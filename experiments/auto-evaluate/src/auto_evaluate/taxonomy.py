from __future__ import annotations

from typing import Any


FAILURE_CODE_DEFINITIONS: dict[str, dict[str, str]] = {
    "TASK_CLASSIFICATION": {
        "label_zh": "任务类型识别错误",
        "description_zh": "误解任务目标、任务类型或题目要求的求解方式。",
    },
    "PARAMETER_EXTRACTION": {
        "label_zh": "参数提取错误",
        "description_zh": "漏读、错读题目输入、固定条件或覆盖参数。",
    },
    "UNIT_CONVERSION": {
        "label_zh": "单位换算错误",
        "description_zh": "单位未统一、换算关系错误或数值与单位不匹配。",
    },
    "TOOL_NOT_CALLED": {
        "label_zh": "应调用工具但未调用",
        "description_zh": "需要工具证据却直接估计、推断或作答。",
    },
    "TOOL_ARGUMENT": {
        "label_zh": "工具参数错误",
        "description_zh": "工具参数缺失、数值错误、名称错误或参数映射错误。",
    },
    "SEARCH_STRATEGY": {
        "label_zh": "搜索与迭代策略错误",
        "description_zh": "边界搜索、候选选择、信息获取顺序或停止策略不合理。",
    },
    "CONSTRAINT_OMISSION": {
        "label_zh": "约束遗漏",
        "description_zh": "未检查题目要求的一个或多个约束。",
    },
    "NUMERICAL_REASONING": {
        "label_zh": "数值推理错误",
        "description_zh": "计算、数值比较、舍入、阈值判断或结果解释错误。",
    },
    "OUTPUT_OMISSION": {
        "label_zh": "必要输出缺失",
        "description_zh": "没有给出题目要求的结果、证据、验证或结论。",
    },
    "OVERCLAIM": {
        "label_zh": "证据不足的过度断言",
        "description_zh": "证据不足、工具未调用或验证不完整，却声称结果已经成立。",
    },
    "ENGINEERING_JUDGMENT": {
        "label_zh": "工程判断错误",
        "description_zh": "数值结果到工程结论、推荐或风险判断的映射不合理。",
    },
    "OTHER": {
        "label_zh": "其他错误",
        "description_zh": "无法归入现有分类的实质性错误。",
    },
    "UNLABELED": {
        "label_zh": "未分类失分",
        "description_zh": "存在评分缺口，但 Judge 没有给出错误类型。",
    },
}

FAILURE_CODES = [code for code in FAILURE_CODE_DEFINITIONS if code != "UNLABELED"]


def failure_code_payload() -> list[dict[str, Any]]:
    return [
        {"code": code, **definition}
        for code, definition in FAILURE_CODE_DEFINITIONS.items()
        if code != "UNLABELED"
    ]


def failure_code_label(code: str) -> str:
    definition = FAILURE_CODE_DEFINITIONS.get(code)
    return definition["label_zh"] if definition else code

