"""Generic ELS regression checks; synthetic evidence only, no live services."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('els09_regressions', ROOT / 'skills/els/runtime.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

SPECS = {'synthetic_plant': dict(required=[], properties={'setting': dict(type='number')}, output_fields={
    'out.cost': dict(metric='cost', unit='USD', basis='USD_2020'),
    'out.flow': dict(metric='flow', unit='m3/s')})}
NATIVE = [dict(type='function', function=dict(name='synthetic_plant', description='Synthetic only',
    parameters=dict(type='object', properties={'setting': dict(type='number')}, additionalProperties=False)))]


def synthetic(name, args):
    return dict(out=dict(cost=1200 + args.get('setting', 0), flow=.2))


class Helpers(unittest.TestCase):
    def setUp(self):
        block = patch('socket.socket.connect', side_effect=AssertionError('Offline only'))
        block.start(); self.addCleanup(block.stop)
        self.rt = engine.ELS('Generic public task; no rubric or answer.', SPECS, NATIVE, synthetic)
        self.act('synthetic_plant', {})

    def act(self, name, args):
        return json.loads(self.rt.execute_tool(name, args))

    def calc(self, **kwargs):
        return self.act('els_calculate', dict(source=dict(record='T001', metric='cost'), **kwargs))

    def test_preserved_08_reproduces_field_description_alias_defect(self):
        old = engine.base.ELS('', SPECS, NATIVE)
        props = old.tool_schemas[1]['function']['parameters']['properties']
        self.assertIs(props['target_unit'], props['to_basis'])
        self.assertEqual(props['target_unit']['description'], props['factor_note']['description'])

    def test_all_schema_objects_are_independent_and_native_schema_unchanged(self):
        self.assertEqual(self.rt.tool_schemas[0], NATIVE[0])
        seen = set()
        def visit(value):
            if isinstance(value, (dict, list)):
                self.assertNotIn(id(value), seen); seen.add(id(value))
                for item in (value.values() if isinstance(value, dict) else value): visit(item)
        visit(self.rt.tool_schemas)
        self.assertEqual(len(self.rt.tool_schemas), 3)

    def test_descriptions_match_operations_and_check_fields(self):
        props = self.rt.tool_schemas[1]['function']['parameters']['properties']
        self.assertIn('Do not supply for convert', props['to_basis']['description'])
        self.assertIn('Optional', props['factor_note']['description'])
        rule = self.rt.tool_schemas[2]['function']['parameters']['properties']['constraints']['items']['properties']
        self.assertIn('threshold', rule['unit']['description'])
        self.assertNotEqual(rule['basis']['description'], props['target_unit']['description'])
        before = props['to_basis']['description']
        props['target_unit']['description'] = 'Changed locally'
        self.assertEqual(props['to_basis']['description'], before)
        self.assertNotEqual(engine.ELS('', SPECS, NATIVE).tool_schemas[1], self.rt.tool_schemas[1])

    def test_legacy_allocator_limitation_reproduced_and_pump_allocation_fixed(self):
        old = engine.base.ELS('', SPECS, NATIVE, synthetic)
        old.execute_tool('synthetic_plant', {})
        rejected = json.loads(old.execute_tool('els_calculate', dict(source=dict(record='T001', metric='cost'),
            operation='scale', factor=1 / 3, target_unit='USD/pump')))
        self.assertIn('error', rejected)
        result = self.calc(operation='ratio', quantity=3, quantity_unit='pump', factor_note='Declared unit count')
        self.assertEqual(result['metric']['value'], 400)
        self.assertEqual(result['metric']['unit'], 'USD/pump')
        self.assertEqual(result['metric']['basis'], 'USD_2020')
        self.assertFalse(result['quantity_declaration']['semantic_verified'])
        self.assertEqual(result['metric']['source_events'], ['T001'])

    def test_independent_quantity_units_and_amounts(self):
        for unit, quantity in [('element', 8), ('m2', 2.5), ('kg', 4), ('train', 2)]:
            with self.subTest(unit=unit):
                result = self.calc(operation='ratio', quantity=quantity, quantity_unit=unit)
                self.assertEqual(result['metric']['value'], 1200 / quantity)
                self.assertEqual(result['metric']['unit'], 'USD/' + unit)

    def test_model_quantity_not_rounded_or_replaced(self):
        result = self.calc(operation='ratio', quantity=3.2, quantity_unit='element')
        self.assertEqual(result['metric']['value'], 375)
        self.assertEqual(result['quantity_declaration']['value'], 3.2)
        self.assertNotIn('complete', result)

    def test_quantity_records_are_checked_and_reused_without_retyping(self):
        result = self.calc(operation='ratio', quantity=3, quantity_unit='pump')
        check = self.act('els_check', dict(records=[result['record']], constraints=[dict(
            metric='value', operator='<=', value=500, unit='USD/pump', basis='USD_2020')]))
        self.assertEqual(check['results'][0]['state'], 'pass')
        again = self.act('els_calculate', dict(source=dict(record=result['record'], metric='value'),
            operation='scale', factor=1.1, to_basis='USD_2025'))
        self.assertAlmostEqual(again['metric']['value'], 440)
        self.assertEqual(again['metric']['unit'], 'USD/pump')
        self.assertEqual(again['metric']['basis'], 'USD_2025')

    def test_invalid_quantities_do_not_create_records(self):
        for quantity in (0, -1, True, float('inf'), float('nan'), '3'):
            with self.subTest(quantity=quantity):
                self.assertIn('error', self.calc(operation='ratio', quantity=quantity, quantity_unit='pump'))
        self.assertEqual(self.rt.derived, {})

    def test_quantity_requires_unit_and_exclusive_divisor(self):
        invalid = [dict(quantity=3), dict(quantity_unit='pump'),
            dict(quantity=3, quantity_unit='USD_2020'), dict(quantity=3, quantity_unit='m3/day'),
            dict(quantity=3, quantity_unit='pump', right=dict(record='T001', metric='cost')),
            dict(quantity=3, quantity_unit='pump', factor=.5),
            dict(quantity=3, quantity_unit='pump', target_unit='USD/pump'),
            dict(quantity=3, quantity_unit='pump', to_basis='USD_2025')]
        for kw in invalid:
            with self.subTest(kw=kw): self.assertIn('error', self.calc(operation='ratio', **kw))
        self.assertIn('error', self.calc(operation='difference', quantity=3, quantity_unit='pump'))
        self.assertEqual(self.rt.derived, {})

    def test_original_evidence_is_unchanged_after_allocation(self):
        before = deepcopy(self.rt.session.export_trace())
        self.calc(operation='ratio', quantity=3, quantity_unit='pump')
        self.assertEqual(self.rt.session.export_trace(), before)

    def test_unconfirmed_scale_cannot_be_laundered_by_quantity_ratio(self):
        first = self.act('els_calculate', dict(source=dict(record='T001', metric='flow'), operation='scale', factor=86400))
        self.assertFalse(first['metric']['verification_eligible'])
        result = self.act('els_calculate', dict(source=dict(record=first['record'], metric='value'),
            operation='ratio', quantity=2, quantity_unit='train'))
        self.assertIn('error', result)

    def test_original_conversion_and_currency_checks_remain(self):
        daily = self.act('els_calculate', dict(source=dict(record='T001', metric='flow'), operation='convert', target_unit='m3/day'))
        self.assertEqual(daily['metric']['value'], 17280)
        intensity = self.calc(operation='ratio', right=dict(record=daily['record'], metric='value'))
        self.assertEqual(intensity['metric']['unit'], 'USD/(m3/day)')
        updated = self.calc(operation='scale', factor=1.2, to_basis='USD_2025')
        self.assertIn('error', self.calc(operation='difference', right=dict(record=updated['record'], metric='value')))
        self.assertIn('error', self.calc(operation='convert', target_unit='USD/pump'))

    def test_generic_quantity_ratio_reaches_model_written_final(self):
        derived = self.calc(operation='ratio', quantity=3, quantity_unit='pump')
        self.act('els_check', dict(records=[derived['record']], constraints=[dict(metric='value',
            operator='<=', value=500, unit='USD/pump', basis='USD_2020')]))
        self.assertEqual(self.rt.accept_final('Model-written conclusion.'), 'Model-written conclusion.')
        self.assertTrue(self.rt.submission['complete'])
        self.assertEqual(self.rt.session._calls, 1)


if __name__ == '__main__': unittest.main()
