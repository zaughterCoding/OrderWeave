"""覆盖代理、预算、故障和共享入口的关键回归。"""
import asyncio
from types import SimpleNamespace

import pytest

from mcp.evidence_coverage import SPEC, extract_needs, retrieve_with_coverage, select_coverage
from mcp.tool_manager import MCPToolManager, Tool


def item(key,content):
    return dict(evidence_id=key,title='',content=content,source_refs=[{'section':'测试'}])


def test_single_need_and_no_evidence_never_fabricate_coverage():
    assert len(extract_needs('退款可以退到哪张银行卡？')) == 1
    assert len(extract_needs('退到礼品卡需要确认卡号吗？')) == 1
    docs=[item('a','完全无关的天气'),item('b','纯粹无关内容')]
    selected,info=select_coverage('提交前是否需要确认，退款可以退到哪张银行卡？',docs,[.8,.7])
    assert selected==[0,1] and set(info['selected_missing'])=={'confirmation','destination'}
    assert info['coverage_is_proxy'] is True


def test_coverage_prefers_new_support_over_duplicate_and_preserves_sources():
    docs=[item('a','展示后需要明确确认'),item('b','展示后需要明确确认'),item('c','退款去向是原支付方式或已有礼品卡')]
    indices,info=select_coverage('确认和退款去向分别有什么要求？',docs,[.9,.85,.4],top_k=2)
    assert indices==[0,2] and info['selected_missing']==[]
    contaminated=[dict(p,query_id='different',qrels={'fake':2},required_evidence_groups=[['fake']]) for p in docs]
    assert select_coverage('确认和退款去向分别有什么要求？',contaminated,[.9,.85,.4],top_k=2)[0]==indices
    with pytest.raises(ValueError): extract_needs({'query_id':'x'})


def test_unconfirmed_product_does_not_create_step_needs():
    assert not extract_needs('PAIR-01和已尝试步骤还要重复吗？',{'scope_state':'ambiguous'})
    packet={'enabled':True,'supported':True,'scope_state':'active'}
    assert len(extract_needs('PAIR-01和已尝试步骤还要重复吗？',packet))>=2


def test_targeted_once_with_bounded_candidates_and_same_scope():
    calls=[]
    initial=[item(str(i),'展示详情后明确确认 '+str(i)) for i in range(20)]
    async def recall(text,k):
        calls.append((text,k))
        if len(calls)==1: return initial+[item('overflow','无关')]
        return [item('wrong','退款去向原支付方式'),*([item('new'+str(i),'退款去向原支付方式') for i in range(12)])]
    async def score(text,docs): return [.8]*len(docs)
    result,diag=asyncio.run(retrieve_with_coverage('确认和退款去向分别有什么要求？',recall,score,
        item_filter=lambda p:p['evidence_id']!='wrong'))
    assert len(calls)==2 and calls[0][1]==20 and calls[1][1]==5
    assert diag['supplement_rounds']==1 and diag['supplement_added']==5
    assert diag['candidate_count']==25 and diag['score_pairs']==25 and len(result)<=5
    assert all(p['source_refs'] and p['content'] and p['evidence_id']!='wrong' for p in result)
    assert diag['selected_missing']==[]


@pytest.mark.parametrize('failure',['empty','duplicate','timeout','score_failure','score_nan'])
def test_supplement_failures_preserve_original_evidence(monkeypatch,failure):
    calls=[]
    async def recall(text,k):
        calls.append(text)
        if len(calls)==1: return [] if failure=='empty' else [item('a','必须确认操作')]
        if failure=='timeout':
            await asyncio.sleep(.05)
        return [item('a','必须确认操作')] if failure=='duplicate' else [item('b','退款去向是原支付方式')]
    async def score(text,docs):
        if docs and docs[0]['evidence_id']=='b':
            if failure=='score_failure': raise RuntimeError('本地评分失败')
            if failure=='score_nan': return [float('nan')]
        return [.8]*len(docs)
    monkeypatch.setitem(SPEC,'supplement_timeout_s',.01)
    result,diag=asyncio.run(retrieve_with_coverage('是否要确认，以及退款去向？',recall,score))
    assert len(calls)<=2 and diag['failure']
    assert [p['evidence_id'] for p in result]==([] if failure=='empty' else ['a'])


def test_two_queries_never_exceed_thirty_candidates():
    calls=[]
    async def recall(text,k):
        calls.append(text)
        prefix=str(len(calls))
        return [item(prefix+'-'+str(i),'没有需求支持的普通材料') for i in range(k+10)]
    async def score(text,docs): return [.6]*len(docs)
    result,diag=asyncio.run(retrieve_with_coverage('确认和退款去向及寄回地址分别是什么？',recall,score))
    assert len(calls)==3 and len(diag['supplemental_queries'])==2
    assert diag['candidate_count']==30 and diag['score_pairs']==30 and len(result)==5
    assert len(diag['selected_missing'])==3


def test_shared_manager_disables_paid_clients_and_scope_cache():
    class NoPaid:
        @property
        def messages(self): raise AssertionError('不得调用付费模型')
    class Local:
        def score(self,query,docs): return [.8]*len(docs)
    manager=MCPToolManager(api_key='test',model='fake',client=NoPaid(),reranker=Local(),retrieval_strategy='targeted')
    seen=[]
    async def handler(params,context):
        seen.append((params,context['knowledge_item_filter']))
        return [item('a','展示后确认'),item('b','退款去向是原支付方式')]
    manager.register(Tool(name='knowledge',description='',schema={},handler=handler,cache_ttl=100))
    async def run():
        for permitted in ('a','b'):
            predicate=lambda p,pid=permitted:p['evidence_id']==pid
            result=await manager.search_with_rewrite('knowledge','模型试图更改的查询',
                original_query='确认和退款去向分别是什么？',item_filter=predicate)
            assert all(p['evidence_id']==permitted for p in result.data)
    asyncio.run(run())
    assert all(x[0]['query']!='模型试图更改的查询' for x in seen)
    assert manager._cache=={}
