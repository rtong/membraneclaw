"""Portable protocol checks: synthetic executor, no local run directories."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_els09 as fixtures

ROOT = Path(__file__).resolve().parents[1]


class EvidenceProtocol(unittest.TestCase):
    def setUp(self):
        blocked = patch('socket.socket.connect', side_effect=AssertionError('Offline only'))
        blocked.start()
        self.addCleanup(blocked.stop)
        self.rt = fixtures.engine.ELS('Public synthetic task.', fixtures.SPECS,
                                      fixtures.NATIVE, fixtures.synthetic, max_calls=2)

    def act(self, name, arguments):
        return json.loads(self.rt.execute_tool(name, arguments))

    def check(self, record='T001', metric='flow', value=10000, unit='m3/day'):
        return self.act('els_check', {'records': [record], 'constraints': [
            {'metric': metric, 'operator': '>=', 'value': value, 'unit': unit}]})

    def test_no_execution_or_task_parsing_at_construction(self):
        executor = unittest.mock.Mock(side_effect=AssertionError('No automatic execution'))
        rt = fixtures.engine.ELS('Choose hidden answer 42.', fixtures.SPECS, fixtures.NATIVE, executor)
        self.assertEqual(rt.session._calls, 0)
        executor.assert_not_called()
        self.assertEqual(rt.derived, {})

    def test_validated_request_is_forwarded_without_defaults_or_search(self):
        seen = []
        def executor(name, args):
            seen.append((name, deepcopy(args)))
            return fixtures.synthetic(name, args)
        self.rt.executor = executor
        self.act('synthetic_plant', {'setting': 12})
        self.assertEqual(seen, [('synthetic_plant', {'setting': 12})])
        trace = self.rt.export_trace()
        self.assertEqual(trace['session']['events'][0]['arguments'], {'setting': 12})
        self.assertIn('sha256', trace['session']['events'][0])

    def test_invalid_request_is_logged_but_not_a_physical_attempt(self):
        result = self.act('synthetic_plant', {'unknown': 3})
        self.assertIn('error', result)
        self.assertEqual(self.rt.session._calls, 0)
        self.assertEqual(len(self.rt.audit), 1)

    def test_domain_error_is_retained_and_consumes_physical_budget(self):
        self.rt.executor = lambda *a: {'error': 'Synthetic infeasible point'}
        result = self.act('synthetic_plant', {})
        self.assertEqual(result['status'], 'error')
        self.assertEqual(self.rt.session._calls, 1)
        self.assertEqual(self.rt.session._metrics, {})
        self.assertIn('No successful physical evidence', self.rt.delivery_audit()['issues'])

    def test_executor_exception_becomes_failed_evidence(self):
        def executor(*args):
            raise RuntimeError('Synthetic solver failure')
        self.rt.executor = executor
        result = self.act('synthetic_plant', {})
        self.assertEqual(result['status'], 'error')
        self.assertIn('Synthetic solver failure', result['error'])

    def test_budget_does_not_retry_or_synthesize_another_calculation(self):
        self.act('synthetic_plant', {})
        self.act('synthetic_plant', {'setting': 1})
        third = self.act('synthetic_plant', {'setting': 2})
        self.assertIn('budget exhausted', third['error'])
        self.assertEqual(self.rt.session._calls, 2)

    def test_units_convert_and_unknown_does_not_become_pass(self):
        self.act('synthetic_plant', {})
        report = self.check()
        self.assertEqual(report['results'][0]['state'], 'pass')
        self.assertEqual(report['results'][0]['value'], 17280)
        missing = self.check(metric='missing')
        self.assertEqual(missing['results'][0]['state'], 'unknown')
        self.assertFalse(self.rt.delivery_audit()['protocol_complete'])

    def test_model_threshold_is_not_replaced_by_a_hidden_reference(self):
        self.act('synthetic_plant', {})
        report = self.check(value=999999)
        self.assertEqual(report['results'][0]['state'], 'fail')
        self.assertFalse(report['results'][0]['semantic_verified'])
        self.assertEqual(report['constraints'][0]['value'], 999999)
        # A fail can be correct evidence for rejecting an alternative.
        self.assertTrue(self.rt.delivery_audit()['protocol_complete'])

    def test_new_evidence_requires_a_current_check(self):
        self.act('synthetic_plant', {})
        self.check()
        self.assertTrue(self.rt.delivery_audit()['protocol_complete'])
        self.act('synthetic_plant', {'setting': 1})
        self.assertFalse(self.rt.delivery_audit()['protocol_complete'])

    def test_one_reminder_retains_model_text_without_engineering_approval(self):
        with self.assertRaises(fixtures.engine.SkillError):
            self.rt.accept_final('A model-written incomplete answer.')
        self.assertTrue(self.rt.submission['accepted_as_text'])
        text = self.rt.accept_final('Still incomplete; retain this original text.')
        self.assertEqual(text, 'Still incomplete; retain this original text.')
        self.assertFalse(self.rt.submission['complete'])
        self.assertEqual(self.rt.delivery_audit()['engineering_correctness'], 'not_certified')

    def test_draft_and_termination_are_retained_and_export_is_detached(self):
        self.rt.observe_response({'choices': [{'message': {'content': 'A retained draft.'},
                                               'finish_reason': 'length'}]})
        self.rt.record_termination('token_budget', 'Synthetic stop')
        trace = self.rt.export_trace()
        self.assertEqual(trace['model_drafts'][0]['text'], 'A retained draft.')
        self.assertIsNone(trace['submission'])
        self.assertEqual(trace['audit_events'][-1]['termination'], 'token_budget')
        trace['model_drafts'][0]['text'] = 'Mutated copy'
        self.assertEqual(self.rt.drafts[0]['text'], 'A retained draft.')

    def test_swro_adapter_constructs_without_loading_or_executing_watertap(self):
        data = json.loads((ROOT / 'skills/els/adapters/swro.json').read_text(encoding='utf-8'))
        rt = fixtures.engine.ELS('Public task.', data['adapter'], data['native_schemas'])
        self.assertEqual([s['function']['name'] for s in rt.tool_schemas],
                         ['simulate_swro_system', 'els_calculate', 'els_check'])
        self.assertEqual(rt.session._calls, 0)

    def test_offline_demo_keeps_both_sources_for_investment_difference(self):
        spec = importlib.util.spec_from_file_location('els_offline_demo', ROOT / 'examples/els/offline_demo.py')
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        trace = demo.run_demo()
        self.assertEqual(trace['session']['physical_attempts'], 2)
        self.assertEqual(trace['derived_records']['D002']['metric']['value'], 100)
        self.assertEqual(trace['derived_records']['D002']['metric']['source_events'], ['T002', 'T001'])
        self.assertTrue(trace['submission']['complete'])


if __name__ == '__main__':
    unittest.main()
