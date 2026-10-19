"""Deterministic integration illustration, NOT a model or physics experiment."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('els_demo_runtime', ROOT / 'skills/els/runtime.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

ADAPTER = {'synthetic_plant': {
    'required': ['setting'], 'properties': {'setting': {'type': 'number'}},
    'output_fields': {'cost': {'metric': 'cost', 'unit': 'USD', 'basis': 'USD_2020'},
                      'flow': {'metric': 'flow', 'unit': 'm3/s'}}}}
SCHEMAS = [{'type': 'function', 'function': {'name': 'synthetic_plant',
    'description': 'Synthetic fixture only; no WaterTAP.',
    'parameters': {'type': 'object', 'properties': {'setting': {'type': 'number'}},
                   'required': ['setting'], 'additionalProperties': False}}}]


def synthetic_executor(name, arguments):
    if name != 'synthetic_plant':
        raise ValueError('Unknown synthetic function')
    return {'cost': 1200 + arguments['setting'], 'flow': 0.2}


def run_demo():
    rt = engine.ELS('Synthetic example, no rubric or hidden answer.',
                    ADAPTER, SCHEMAS, synthetic_executor)
    # These actions illustrate host wiring. In real use they come from the LLM.
    actions = [
        ('synthetic_plant', {'setting': 0}),
        ('synthetic_plant', {'setting': 300}),
        ('els_calculate', {'source': {'record': 'T002', 'metric': 'cost'},
            'operation': 'difference', 'right': {'record': 'T001', 'metric': 'cost'}}),
        ('els_calculate', {'source': {'record': 'D001', 'metric': 'value'},
            'operation': 'ratio', 'quantity': 3, 'quantity_unit': 'pump'}),
        ('els_check', {'records': ['D002'], 'constraints': [{'metric': 'value',
            'operator': '<=', 'value': 120, 'unit': 'USD/pump', 'basis': 'USD_2020'}]}),
    ]
    for name, args in actions:
        feedback = json.loads(rt.execute_tool(name, args))
        if 'error' in feedback:
            raise RuntimeError(feedback['error'])
    rt.accept_final('Illustration text: the declared increment is 100 USD/pump, within 120 USD/pump. '
                    'This text is a fixture, not a model decision.')
    return rt.export_trace()


if __name__ == '__main__':
    trace = run_demo()
    print(json.dumps({'demo': 'synthetic only',
        'physical_fixture_attempts': trace['session']['physical_attempts'],
        'derived': trace['derived_records'], 'delivery_audit': trace['delivery_audit'],
        'model_physics_judge_experiments_started': 0}, ensure_ascii=False, indent=2))
