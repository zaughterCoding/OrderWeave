"""固定预算的词法覆盖代理与一次补检；不判断答案正确性或交易资格。"""
from __future__ import annotations

import asyncio
import re
import time

from mcp.hybrid_retrieval import tokenize
from mcp.qwen_reranker import validate_scores

SPEC = dict(initial_k=20, supplement_rounds=1, supplement_queries=2,
            supplement_k=5, new_candidates=10, total_candidates=30, final_k=5,
            max_needs=6, coverage_weight=0.7, duplicate_penalty=0.15,
            retrieve_timeout_s=20, score_timeout_s=60, supplement_timeout_s=30)
STRATEGIES = ('relevance', 'pool', 'coverage', 'targeted', 'deeper')

# 仅从用户明示的不同概念取需求；文档模式只是可解释的支持线索。
NEED_RULES = (
    ('conditions', '退货条件', r'退货条件|资格|能退|能.{0,6}申请|订单状态|delivered|processed|pending', r'delivered|pending|processed', False),
    ('confirmation', '提交前确认', r'确认|同意|提交前|没提交|尚未提交|改.{0,5}(数量|商品)', r'确认|明确同意|展示', False),
    ('destination', '退款去向', r'退款方式|支付方式|退款去向|银行卡|信用卡|收款|礼品卡|paypal|原路|退到', r'原支付|已有.{0,8}礼品卡|退款去向|收款', False),
    ('status', '申请状态与到账边界', r'已申请|申请成功|到账|收到退款|入账|return requested', r'return requested|申请已提交|已申请退货|到账|入账', False),
    ('attempts', '已尝试步骤与用户反馈', r'已.{0,6}(试|配对)|已尝试|试过|还.{0,4}不行|未解决|排查.{0,4}失败|尝试记录', r'用户报告|工作记录|已尝试|用户.{0,4}反馈', True),
    ('pairing', '蓝牙配对步骤', r'pair-0[123]|开关|供电|断开|旧记录|重新配对|重配', r'pair-0[123]|供电|其他设备|旧记录', True),
    ('stop', '停止重复排查', r'停止|重复|反复', r'停止|重复', True),
    ('handoff', '人工交接边界', r'人工|交接', r'摘要|模拟交接|真实坐席|真实工单', False),
    ('privacy', '个人资料与恢复设置边界', r'私人|个人.{0,4}(资料|文件)|备份|恢复设置|清除|清空|密码', r'个人资料|备份|清除数据|恢复设置|密码', False),
    ('parts', '商品与配件核对', r'配件|部件|颜色|尺码|清点|实物', r'配件|部件|颜色|尺码|清点', False),
    ('shipping', '寄回资料边界', r'寄回地址|仓库地址|运费|承运|快递|寄回.{0,4}地址', r'寄回地址|承运|运费', False),
    ('repair', '拆机维修边界', r'拆机|维修|修理', r'拆机|维修|拆卸', False),
)


def extract_needs(query, context=None):
    if not isinstance(query, str):
        raise ValueError('原始问题必须是文本')
    packet = context if isinstance(context, dict) else {}
    supported = (packet.get('enabled') is True and packet.get('supported') is True
                 and packet.get('scope_state') == 'active')
    clauses = re.split(r'[，。；？！?\n]', query)
    needs = []
    for key, label, user_pattern, doc_pattern, support_only in NEED_RULES:
        match = re.search(user_pattern, query, re.I)
        if match is None or support_only and not supported:
            continue
        clause = next((c.strip() for c in clauses if re.search(user_pattern, c, re.I)), '')
        needs.append(dict(key=key, label=label, doc_pattern=doc_pattern,
                          query=(label + '：' + clause[:180]), position=match.start()))
    needs = sorted(needs, key=lambda n: n['position'])[:SPEC['max_needs']]
    if len(needs) > 1 and not re.search(r'和|与|以及|同时|另外|分别|还有|并|但|又|除了|[，,；;、]',query):
        return needs[:1]
    return needs


def support_map(needs, items):
    return [{n['key'] for n in needs if re.search(n['doc_pattern'],
             str(p.get('title', '')) + '\n' + str(p.get('content', '')), re.I)} for p in items]


def _union(sets):
    return set().union(*sets) if sets else set()


def select_coverage(query, items, scores, top_k=5, context=None):
    """在同一整体相关性评分上贪心选择；代理无增益时原序回退。"""
    scores = validate_scores(scores, len(items))
    k = min(max(1, top_k), SPEC['final_k'])
    ranked = sorted(range(len(items)), key=lambda i: (-scores[i], i))
    needs = extract_needs(query, context)
    support = support_map(needs, items)
    base = ranked[:k]
    base_covered = _union([support[i] for i in base])
    all_keys = {n['key'] for n in needs}
    info = dict(needs=[{k:v for k,v in n.items() if k != 'doc_pattern'} for n in needs],
                coverage_is_proxy=True, pool_missing=sorted(all_keys - _union(support)))
    if len(needs) < 2 or base_covered == all_keys:
        return base, dict(info, status='relevance_fallback', selected_missing=sorted(all_keys-base_covered))
    tokens = [set(tokenize(str(p.get('title', ''))+' '+str(p.get('content', '')))) for p in items]
    selected, covered = [], set()
    ranks = {index:rank+1 for rank,index in enumerate(ranked)}
    while len(selected) < min(k, len(items)):
        def gain(i):
            duplicate = max((len(tokens[i] & tokens[j])/max(len(tokens[i] | tokens[j]),1)
                             for j in selected), default=0)
            return (1/(1+ranks[i]/5) + SPEC['coverage_weight']*len(support[i]-covered)/len(needs)
                    - SPEC['duplicate_penalty']*duplicate)
        best = max((i for i in ranked if i not in selected), key=lambda i:(gain(i),-ranks[i]))
        selected.append(best)
        covered.update(support[best])
    if len(covered) <= len(base_covered):
        selected, covered = base, base_covered
        status = 'no_effective_gain'
    else:
        status = 'coverage_applied'
    return selected, dict(info, status=status, selected_missing=sorted(all_keys-covered))


def _key(item):
    return str(item.get('evidence_id') or item.get('id') or item.get('content') or '')


async def retrieve_with_coverage(query, retrieve, score, *, strategy='targeted',
                                 top_k=5, context=None, item_filter=None):
    """所有额外召回沿用同一个过滤器，失败时保留此前有效结果。"""
    if strategy not in STRATEGIES[1:]:
        raise ValueError('未知实验策略')
    started = time.perf_counter()
    diag = dict(strategy=strategy, spec=SPEC, queries=[], score_pairs=0, score_calls=0,
                supplement_rounds=0, supplement_added=0, supplemental_queries=[],
                failure='', coverage_is_proxy=True)
    diag['selected_missing'] = [n['key'] for n in extract_needs(query,context)]
    items, scores = [], []

    async def recall(text, k):
        diag['queries'].append(text)
        found = await asyncio.wait_for(retrieve(text, k), SPEC['retrieve_timeout_s'])
        unique, seen = [], set()
        for item in found:
            if (not isinstance(item, dict) or not str(item.get('content') or '').strip()
                    or item_filter is not None and not item_filter(item)):
                continue
            key = _key(item)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique[:k]

    async def scoring(batch):
        diag['score_pairs'] += len(batch)
        diag['score_calls'] += bool(batch)
        return validate_scores(await asyncio.wait_for(score(query, batch), SPEC['score_timeout_s']),len(batch)) if batch else []

    def choose():
        indices, info = select_coverage(query, items, scores, top_k, context)
        if strategy == 'pool':
            indices = sorted(range(len(items)), key=lambda i:(-scores[i],i))[:min(top_k,5)]
            needs = extract_needs(query,context)
            info['selected_missing'] = sorted({n['key'] for n in needs}-_union([support_map(needs,items)[i] for i in indices]))
            info['status'] = 'relevance_pool'
        return indices, info

    try:
        items = await recall(query,SPEC['initial_k'])
        diag['initial_candidates'] = [dict(p) for p in items]
        if not items:
            diag['failure'] = 'empty_initial_pool'
            return [], diag
        scores = await scoring(items)
        indices, info = choose()
        diag['initial_selection'] = dict(info)
        needs = extract_needs(query,context)
        gaps = [n for n in needs if n['key'] in info['pool_missing']]
        if strategy in ('targeted','deeper') and len(needs) >= 2 and gaps:
            diag['supplement_rounds'] = 1
            deadline = time.perf_counter() + SPEC['supplement_timeout_s']
            for need in gaps[:SPEC['supplement_queries']]:
                if time.perf_counter() >= deadline:
                    diag['failure'] = 'supplement_timeout'
                    break
                text = need['query'] if strategy == 'targeted' else query
                diag['supplemental_queries'].append(text)
                k = SPEC['supplement_k'] if strategy == 'targeted' else min(30,20+SPEC['supplement_k']*(len(diag['supplemental_queries'])))
                found = await asyncio.wait_for(recall(text,k),max(0.001,deadline-time.perf_counter()))
                seen = {_key(p) for p in items}
                fresh = [p for p in found if _key(p) not in seen][:min(SPEC['new_candidates']-diag['supplement_added'],SPEC['total_candidates']-len(items))]
                if not fresh:
                    diag['failure'] = 'no_new_evidence'
                    break
                # 新批评分成功后才替换结果，超时不丢弃旧证据。
                new_scores = await asyncio.wait_for(scoring(fresh),max(0.001,deadline-time.perf_counter()))
                items.extend(fresh)
                scores.extend(new_scores)
                diag['supplement_added'] += len(fresh)
            indices, info = choose()
        diag.update(info)
        result = [dict(items[i],rerank_score=scores[i]) for i in indices]
        return result, diag
    except Exception as exc:
        diag['failure'] = type(exc).__name__+': '+str(exc)[:180]
        if scores:
            indices, info = choose()
            diag.update(info)
            return [dict(items[i],rerank_score=scores[i]) for i in indices], diag
        return items[:min(top_k,5)], diag
    finally:
        diag['candidate_count'] = len(items)
        diag['candidates'] = [dict(p) for p in items]
        diag['scores'] = list(scores)
        diag['relevance_order'] = [_key(items[i]) for i in sorted(range(len(scores)),key=lambda i:(-scores[i],i))]
        diag['total_ms'] = (time.perf_counter()-started)*1000
