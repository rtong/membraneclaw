"""ELS 0.9: independent schema descriptions and quantity-normalized evidence.

Derived from the preserved predecessor; no task lookup, candidate generation or automatic rounding.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

from auto_evaluate.engineering_skills.primitives import SkillError, number

spec = importlib.util.spec_from_file_location('_els08_preserved_helpers', Path(__file__).with_name('_arithmetic.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class ELS(base.ELS):
    protocol_version = 'evidence-linked-skills@0.9.0'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session._contract['execution_origin'] = 'ELS_0.9_native_executor'
        # deepcopy preserves shared dictionary aliases. JSON detaches every
        # schema occurrence before assigning field-specific descriptions.
        self.tool_schemas = json.loads(json.dumps(self.tool_schemas))
        helpers = {s['function']['name']: s['function'] for s in self.tool_schemas}
        calc = helpers['els_calculate']
        props = calc['parameters']['properties']
        props['target_unit']['description'] = 'Output unit for convert, or explicitly unit-preserving/compatible physical scale. Not a currency year or quantity allocation.'
        props['to_basis']['description'] = 'Only for scale: new currency-year basis, exactly USD_YYYY. Put explanations in factor_note. Do not supply for convert.'
        props['factor_note']['description'] = 'Optional explanatory text for ANY operation; changes no arithmetic and certifies no task interpretation.'
        props['factor']['description'] = 'Only for scale: finite numeric multiplier. Unit conversion should use convert.'
        props['quantity'] = dict(type='number', exclusiveMinimum=0,
            description='Only for ratio without right: positive model-declared denominator quantity. Choose the quantity and any required rounding yourself.')
        props['quantity_unit'] = dict(type='string',
            description='Required with quantity: one nonmonetary unit label, e.g. pump, element, m2 or kg. No currency basis, formula or compound unit.')
        for name in ['source', 'right']:
            props[name]['properties']['record']['description'] = 'Exact returned T/D record ID.'
            props[name]['properties']['metric']['description'] = 'Returned/native metric name for T records; exactly value for D records.'
        calc['description'] = 'Read saved evidence; convert, scale or compare. ratio divides by either right={record,metric} OR a model-declared quantity plus quantity_unit. Returns explicit units and provenance; never selects the denominator or interprets the task.'
        check = helpers['els_check']['parameters']['properties']
        check['records']['items']['description'] = 'Exact returned T/D record ID to check.'
        rule = check['constraints']['items']['properties']
        rule['metric']['description'] = 'Returned/native metric name for T records; exactly value for D records.'
        rule['unit']['description'] = 'Unit of the model-declared threshold; must be compatible with the recorded metric.'
        rule['basis']['description'] = 'For monetary evidence, explicit currency-year basis such as USD_2020; not a unit or explanatory sentence.'

    def _calculate(self, source, operation, right=None, target_unit=None, factor=None,
                   to_basis=None, factor_note=None, quantity=None, quantity_unit=None):
        if quantity is None and quantity_unit is None:
            return super()._calculate(source, operation, right=right, target_unit=target_unit,
                factor=factor, to_basis=to_basis, factor_note=factor_note)
        if operation != 'ratio':
            raise SkillError('quantity and quantity_unit are only for ratio')
        if right is not None:
            raise SkillError('ratio uses either right OR quantity with quantity_unit, never both')
        if any(v is not None for v in [target_unit, factor, to_basis]):
            raise SkillError('Quantity ratio does not use target_unit, factor or to_basis; the output unit is derived automatically')
        if quantity is None or number(quantity) <= 0:
            raise SkillError('quantity must be a positive finite number')
        if not isinstance(quantity_unit, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', quantity_unit):
            raise SkillError('quantity_unit must be a single nonmonetary unit label; use recorded right evidence for compound-unit division')
        if quantity_unit.startswith('USD'):
            raise SkillError('Monetary denominators require recorded right evidence with its currency basis')
        metric = self._metric(source)
        numerator = metric['unit']
        output_unit = (numerator if '/' not in numerator else '(' + numerator + ')') + '/' + quantity_unit
        declaration = dict(value=quantity, unit=quantity_unit, origin='model_declared', semantic_verified=False)
        metric.update(value=number(metric['value'] / quantity), unit=output_unit,
                      quantity_declaration=deepcopy(declaration))
        record = 'D%03d' % (len(self.derived) + 1)
        result = dict(record=record, metric=metric, operation=operation, source=deepcopy(source),
                      right=None, quantity_declaration=declaration, annotation=factor_note,
                      requested_operation=operation)
        self.derived[record] = deepcopy(result)
        self.revision += 1
        return result
