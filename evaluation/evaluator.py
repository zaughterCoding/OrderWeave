"""端到端的助手评测。

意图识别用准确率和 macro-F1 衡量。对话质量由 LLM Judge 打分。Judge 失败
记录为评分器错误，绝不进入分数均值或基线文件。
"""
import json
import logging
import pathlib
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.intent_recognizer import IntentRecognizer
from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


@dataclass
class IntentTestCase:
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


@dataclass
class QualityScores:
    """单个响应的 LLM-as-Judge 评分。"""

    relevance: float
    accuracy: float
    completeness: float
    helpfulness: float
    judge_failed: bool = False
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        if self.judge_failed:
            return 0.0
        return statistics.mean(
            [
                self.relevance,
                self.accuracy,
                self.completeness,
                self.helpfulness,
            ]
        )


@dataclass
class EvalResult:
    test_id: str
    passed: bool
    scores: Dict[str, float]
    detail: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    timestamp: str
    total: int
    passed: int
    pass_rate: float
    avg_scores: Dict[str, float]
    regressions: List[str]
    recommendations: List[str]
    results: List[EvalResult]
    grader_errors: int = 0


class LLMJudge:
    """从相关性、准确性、完整性和有用性四个维度给响应打分。"""

    JUDGE_PROMPT = """你是一个客服质量评估专家。请对以下客服响应进行评分。

用户问题: {question}
Agent 响应: {response}
{context_section}

请从以下四个维度评分（0.0-1.0），返回 JSON：
- relevance: 响应是否直接针对用户问题（0=完全无关，1=完全相关）
- accuracy: 信息是否准确无误（0=明显错误，1=完全正确）
- completeness: 是否完整解决了用户需求（0=完全没解决，1=完全解决）
- helpfulness: 用户能否据此采取行动（0=毫无帮助，1=非常有帮助）

只返回 JSON，例如: {{"relevance": 0.9, "accuracy": 0.8, "completeness": 0.7, "helpfulness": 0.85}}"""

    REQUIRED_FIELDS = ("relevance", "accuracy", "completeness", "helpfulness")

    def __init__(self, client: AsyncAnthropic, model: str):
        if not model:
            raise ValueError("model is required")
        self._client = client
        self._model = model

    async def judge(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
    ) -> QualityScores:
        ctx_section = f"背景信息: {context}" if context else ""
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            response=response,
            context_section=ctx_section,
        )
        prompt = self._clean_text(prompt)
        try:
            resp = await self._client.messages.create(
                model=self._model,
                # 结构化 JSON 输出需要给推理留出空间，1024 可覆盖较长答复。
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("judge response contains no JSON object")
            data = json.loads(raw[start:end])
            if not isinstance(data, dict):
                raise ValueError("judge response JSON is not an object")

            values: Dict[str, float] = {}
            for name in self.REQUIRED_FIELDS:
                if name not in data:
                    raise ValueError(f"judge response missing {name}")
                raw_value = data[name]
                if isinstance(raw_value, bool):
                    raise ValueError(f"judge score {name} is not numeric")
                value = float(raw_value)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"judge score {name} is outside 0-1")
                values[name] = value

            return QualityScores(**values)
        except Exception as ex:
            logger.warning(f"LLM Judge 失败: {ex}")
            return QualityScores(
                0.0,
                0.0,
                0.0,
                0.0,
                judge_failed=True,
                error=str(ex),
            )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """发送 Judge prompt 前移除代理项字符。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")


class IntentEvaluator:
    """衡量意图识别的准确率和 macro-F1。"""

    def __init__(self, recognizer: IntentRecognizer):
        self._recognizer = recognizer

    async def evaluate(self, cases: List[IntentTestCase]) -> Dict[str, Any]:
        predictions, ground_truth = [], []
        case_details: List[Dict[str, Any]] = []

        for case in cases:
            result = await self._recognizer.recognize(case.message)
            predicted = result.intent.value
            predictions.append(predicted)
            ground_truth.append(case.expected_intent)
            case_details.append(
                {
                    "message": case.message,
                    "expected": case.expected_intent,
                    "predicted": predicted,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                }
            )

        correct = sum(
            predicted == expected
            for predicted, expected in zip(predictions, ground_truth)
        )
        accuracy = correct / len(predictions) if predictions else 0.0

        labels = sorted(set(ground_truth + predictions))
        per_class: Dict[str, Dict[str, float]] = {}
        for label in labels:
            tp = sum(
                predicted == label and expected == label
                for predicted, expected in zip(predictions, ground_truth)
            )
            fp = sum(
                predicted == label and expected != label
                for predicted, expected in zip(predictions, ground_truth)
            )
            fn = sum(
                predicted != label and expected == label
                for predicted, expected in zip(predictions, ground_truth)
            )
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            per_class[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        macro_f1 = (
            statistics.mean(value["f1"] for value in per_class.values())
            if per_class
            else 0.0
        )

        return {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": per_class,
            "total": len(cases),
            "correct": correct,
            "cases": case_details,
        }


class EndToEndEvaluator:
    """运行意图与对话评测，并分别保存报告。"""

    PASS_THRESHOLD = 0.75

    def __init__(
        self,
        orchestrator,
        recognizer: Optional[IntentRecognizer],
        api_key: str = "",
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        baseline_path: Optional[str] = None,
        report_dir: Optional[str] = None,
        judge: Optional[LLMJudge] = None,
    ):
        if not model:
            raise ValueError("model is required")

        if judge is None:
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = AsyncAnthropic(**kwargs)
            judge = LLMJudge(client, model)

        self._orchestrator = orchestrator
        self._judge = judge
        self._intent_evaluator = IntentEvaluator(recognizer) if recognizer else None
        self._history: List[EvalReport] = []
        self._baseline_path = pathlib.Path(baseline_path) if baseline_path else None
        self._report_dir = pathlib.Path(report_dir) if report_dir else None
        if self._report_dir is None and self._baseline_path is not None:
            self._report_dir = self._baseline_path.parent / "reports"
        self._baseline: Optional[EvalReport] = self._load_baseline()

    async def run(
        self,
        intent_cases: Optional[List[IntentTestCase]] = None,
        dialog_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> EvalReport:
        """运行配置的评测用例并返回一份报告。"""
        results: List[EvalResult] = []
        all_scores: Dict[str, List[float]] = {
            "relevance": [],
            "accuracy": [],
            "completeness": [],
            "helpfulness": [],
        }

        intent_metrics: Dict[str, Any] = {}
        if intent_cases:
            if self._intent_evaluator is None:
                raise ValueError("recognizer is required for intent evaluation")
            intent_metrics = await self._intent_evaluator.evaluate(intent_cases)
            passed = intent_metrics["accuracy"] >= self.PASS_THRESHOLD
            results.append(
                EvalResult(
                    test_id="intent_recognition",
                    passed=passed,
                    scores={
                        "accuracy": intent_metrics["accuracy"],
                        "macro_f1": intent_metrics["macro_f1"],
                    },
                    detail=(
                        f"准确率 {intent_metrics['accuracy']:.1%}，"
                        f"Macro-F1 {intent_metrics['macro_f1']:.3f}"
                    ),
                    metadata={
                        "total": intent_metrics.get("total", 0),
                        "correct": intent_metrics.get("correct", 0),
                        "cases": intent_metrics.get("cases", []),
                    },
                )
            )

        if dialog_cases:
            for index, case in enumerate(dialog_cases):
                case_results = await self._evaluate_dialog_case(case, index)
                results.extend(case_results)
                for result in case_results:
                    for key in all_scores:
                        if key in result.scores:
                            all_scores[key].append(result.scores[key])

        avg_scores = {
            key: round(statistics.mean(values), 4)
            for key, values in all_scores.items()
            if values
        }
        if intent_metrics:
            avg_scores["intent_accuracy"] = intent_metrics["accuracy"]

        passed_count = sum(1 for result in results if result.passed)
        pass_rate = passed_count / len(results) if results else 0.0
        grader_errors = sum(
            1 for result in results if "GRADER_ERROR" in result.metadata
        )

        regressions = self._detect_regressions(avg_scores)
        recommendations = self._recommendations(avg_scores, grader_errors)

        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            total=len(results),
            passed=passed_count,
            pass_rate=round(pass_rate, 4),
            avg_scores=avg_scores,
            regressions=regressions,
            recommendations=recommendations,
            results=results,
            grader_errors=grader_errors,
        )
        self._history.append(report)
        self._save_report(report)
        return report

    async def _evaluate_dialog_case(
        self, case: Dict[str, Any], case_idx: int
    ) -> List[EvalResult]:
        """评测一个单轮或多轮对话用例。"""
        from agents.agent_orchestrator import Request as OrcRequest

        questions = self._dialog_turns(case)
        if not questions:
            return []

        conv_id = str(case.get("conv_id") or f"eval_{case_idx}")
        user_id = str(case.get("user_id") or "eval_user")
        history: List[Dict[str, str]] = []
        results: List[EvalResult] = []

        for turn_idx, question in enumerate(questions):
            context = self._history_context(history)
            orch_request = OrcRequest(
                message=question,
                user_id=user_id,
                conv_id=conv_id,
                context=context,
                history=history[-6:] if history else None,
            )
            orch_result = await self._orchestrator.run(orch_request)
            actual_answer = orch_result.response

            scores = await self._judge.judge(
                question, actual_answer, context=context or None
            )

            metadata = {
                "question": question,
                "response": actual_answer,
                "agent_type": orch_result.agent_type.value,
                "intent": (
                    orch_result.intent.value if orch_result.intent else None
                ),
                "turn": turn_idx,
                "conv_id": conv_id,
                "judge_failed": scores.judge_failed,
                "judge_error": scores.error,
            }

            if scores.judge_failed:
                passed = False
                result_scores: Dict[str, float] = {}
                error = scores.error or "judge_failed"
                metadata["GRADER_ERROR"] = error
                detail = f"Q: {question[:30]}... → 评分失败: {error}"
            else:
                passed = scores.overall >= self.PASS_THRESHOLD
                result_scores = {
                    "relevance": scores.relevance,
                    "accuracy": scores.accuracy,
                    "completeness": scores.completeness,
                    "helpfulness": scores.helpfulness,
                    "overall": scores.overall,
                }
                detail = (
                    f"Q: {question[:30]}... → 综合评分 {scores.overall:.3f}"
                )

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": actual_answer})

            test_id = (
                f"dialog_{case_idx}"
                if len(questions) == 1
                else f"dialog_{case_idx}_turn_{turn_idx}"
            )
            results.append(
                EvalResult(
                    test_id=test_id,
                    passed=passed,
                    scores=result_scores,
                    detail=detail,
                    metadata=metadata,
                )
            )

        return results

    @staticmethod
    def _dialog_turns(case: Dict[str, Any]) -> List[str]:
        turns = case.get("turns")
        if isinstance(turns, list):
            return [str(turn) for turn in turns if str(turn).strip()]
        question = case.get("question")
        return [str(question)] if question else []

    @staticmethod
    def _history_context(history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = [
            f"{message['role']}: {message['content']}"
            for message in history[-8:]
        ]
        return "[评测多轮历史]\n" + "\n".join(lines)

    def _detect_regressions(self, current: Dict[str, float]) -> List[str]:
        """与只读基线比较；没有基线时与上一次运行比较。"""
        previous = self._baseline
        if previous is None and self._history:
            previous = self._history[-1]
        if previous is None:
            return []

        regressions = []
        for metric, value in current.items():
            if metric in previous.avg_scores and previous.avg_scores[metric] > 0:
                delta = (
                    (value - previous.avg_scores[metric])
                    / previous.avg_scores[metric]
                )
                if delta < -0.05:
                    regressions.append(
                        f"{metric}: {previous.avg_scores[metric]:.3f} → {value:.3f} "
                        f"(退化 {abs(delta):.1%})"
                    )
        return regressions

    def _recommendations(
        self,
        scores: Dict[str, float],
        grader_errors: int,
    ) -> List[str]:
        recs = []
        if grader_errors:
            recs.append(
                f"{grader_errors} 条结果因 Judge 评分失败未计入均值，"
                "请检查 Judge 输出格式与模型配置"
            )
        metric_count = len(recs)
        if scores.get("intent_accuracy", 1.0) < 0.90:
            recs.append(
                "意图识别准确率 < 90%：增加 Few-shot 示例，或对低 F1 的意图类别补充训练数据"
            )
        if scores.get("relevance", 1.0) < 0.75:
            recs.append(
                "相关性偏低：检查 Agent system_prompt，确保 Agent 聚焦于用户问题"
            )
        if scores.get("completeness", 1.0) < 0.75:
            recs.append(
                "完整性偏低：Agent 可能过早结束回答，考虑在 prompt 中要求提供完整解决方案"
            )
        if scores.get("helpfulness", 1.0) < 0.75:
            recs.append(
                "有用性偏低：回答可能过于抽象，考虑要求 Agent 提供具体操作步骤"
            )
        quality_metrics = (
            "intent_accuracy",
            "relevance",
            "completeness",
            "helpfulness",
        )
        if not any(key in scores for key in quality_metrics):
            recs.append("没有可用的评分结果，无法判断质量趋势")
        elif len(recs) == metric_count:
            recs.append("所有指标均达标，继续保持")
        return recs

    @property
    def history(self) -> List[EvalReport]:
        return self._history

    def _load_baseline(self) -> Optional[EvalReport]:
        if not self._baseline_path or not self._baseline_path.exists():
            return None
        try:
            data = json.loads(
                self._baseline_path.read_text(encoding="utf-8")
            )
            return self._report_from_dict(data)
        except Exception as ex:
            logger.warning(f"读取评测基线失败: {ex}")
            return None

    def _save_report(self, report: EvalReport) -> None:
        """写入新的报告文件，不改动基线。"""
        if self._report_dir is None:
            return
        try:
            self._report_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = self._report_dir / f"report_{stamp}.json"
            counter = 1
            while path.exists():
                path = self._report_dir / f"report_{stamp}_{counter}.json"
                counter += 1
            path.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as ex:
            logger.warning(f"保存评测报告失败: {ex}")

    @staticmethod
    def _report_from_dict(data: Dict[str, Any]) -> EvalReport:
        return EvalReport(
            timestamp=data.get("timestamp", ""),
            total=int(data.get("total", 0)),
            passed=int(data.get("passed", 0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            avg_scores=dict(data.get("avg_scores", {})),
            regressions=list(data.get("regressions", [])),
            recommendations=list(data.get("recommendations", [])),
            results=[
                EvalResult(
                    test_id=result.get("test_id", ""),
                    passed=bool(result.get("passed", False)),
                    scores=dict(result.get("scores", {})),
                    detail=result.get("detail", ""),
                    metadata=dict(result.get("metadata", {})),
                )
                for result in data.get("results", [])
            ],
            grader_errors=int(data.get("grader_errors", 0)),
        )


DEFAULT_INTENT_CASES: List[IntentTestCase] = [
    IntentTestCase("我的订单什么时候到？", "logistics"),
    IntentTestCase("帮我取消订单", "request"),
    IntentTestCase("你们服务太差了！", "complaint"),
    IntentTestCase("应用一直报500错误", "technical_crash"),
    IntentTestCase("为什么扣了两次款？", "payment_issue"),
    IntentTestCase("我要投诉，转人工！", "human_handoff"),
    IntentTestCase("你好", "greeting"),
    IntentTestCase("修改我的邮箱地址", "account"),
    IntentTestCase("帮我开发票", "invoice"),
    IntentTestCase("退款多久到账？", "refund"),
    IntentTestCase("登录一直报401", "technical_login"),
]

DEFAULT_DIALOG_CASES: List[Dict[str, Any]] = [
    {"question": "我的订单 #12345 还没到，已经超时了"},
    {"question": "应用登录一直报错 401"},
    {"question": "为什么这个月多扣了 50 块钱？"},
    {"question": "帮我把收货地址改成北京市朝阳区"},
    {"turns": ["你好，我想退款", "订单号是 #12345", "退款多久能到账？"]},
]
