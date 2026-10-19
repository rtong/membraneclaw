"""Evidence operations used by the current ELS runtime."""
from copy import deepcopy
import importlib.util
from pathlib import Path

from auto_evaluate.engineering_skills.primitives import SkillError, convert, number

spec = importlib.util.spec_from_file_location('_els07_preserved_core', Path(__file__).with_name('_core.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class ELS(base.ELS):
    protocol_version = 'evidence-linked-skills@0.8.0'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session._contract['execution_origin'] = 'ELS_0.8_native_executor'
        schema = next(s['function'] for s in self.tool_schemas if s['function']['name'] == 'els_calculate')
        props = schema['parameters']['properties']
        props['factor_note']['description'] = 'Optional explanatory annotation for ANY operation; does not affect arithmetic.'
        props['target_unit']['description'] = 'Required for convert; for scale explicitly state the intended output unit. Unknown unit intent cannot pass a constraint check.'
        schema['description'] = 'Read evidence; convert, scale or compare. ratio supports different dimensions (e.g. investment / daily capacity) and returns explicit composite units. No task interpretation is certified.'

    def _metric(self, reference):
        metric = super()._metric(reference)
        if metric.get('verification_eligible') is False:
            raise SkillError('Unit intent is unresolved in ' + reference['record'] +
                '. Recalculate from its ORIGINAL source with operation=convert and target_unit for a unit change, '
                'or scale with an explicit intended target_unit. Do not convert the already-scaled value again.')
        return metric

    def _calculate(self, source, operation, right=None, target_unit=None, factor=None, to_basis=None, factor_note=None):
        allowed = {'convert': {'target_unit'}, 'scale': {'factor', 'target_unit', 'to_basis'},
                   'difference': {'right'}, 'ratio': {'right'}, 'relative_difference': {'right'}}[operation]
        supplied = {k for k, v in dict(right=right, target_unit=target_unit, factor=factor, to_basis=to_basis).items() if v is not None}
        invalid = supplied - allowed
        if invalid:
            raise SkillError(operation + ' does not use fields ' + ', '.join(sorted(invalid)) +
                             '; allowed numeric fields: ' + ', '.join(sorted(allowed)) + '. factor_note is optional for every operation.')
        metric = self._metric(source)
        result = None
        if operation == 'scale' and target_unit is not None:
            target, basis = base.unit_basis(target_unit)
            if basis and basis != metric.get('basis'):
                raise SkillError('Currency-year changes need to_basis and an explicit adjustment factor, not a unit conversion.')
            if target != metric['unit']:
                expected = convert(1, metric['unit'], target)
                import math
                if factor is None or not math.isclose(number(factor), expected, rel_tol=1e-12, abs_tol=0):
                    raise SkillError('Wrong unit-conversion factor; use operation=convert with target_unit=' + target)
                if to_basis is not None:
                    raise SkillError('A unit conversion cannot also change currency basis')
                result = super()._calculate(source, 'convert', target_unit=target_unit)
                result['normalization'] = 'Explicit target unit and factor agree dimensionally; recorded as convert.'
        if operation == 'ratio':
            if right is None:
                raise SkillError('ratio requires right={record,metric}')
            other = self._metric(right)
            if other['value'] == 0:
                raise SkillError('Cannot divide by zero evidence')
            try:
                rv = convert(other['value'], other['unit'], metric['unit'])
            except SkillError:
                rv = None
            if rv is not None:
                if metric.get('basis') != other.get('basis'):
                    raise SkillError('Ratios of compatible dimensions require matching currency/physical bases')
                metric.update(value=metric['value'] / rv, unit='fraction', basis=None)
            else:
                lb, rb = metric.get('basis'), other.get('basis')
                if lb and rb and lb != rb:
                    raise SkillError('Different currency bases cannot be cancelled in a ratio')
                numerator_unit = metric['unit']
                metric.update(value=metric['value'] / other['value'],
                              unit=numerator_unit + '/(' + other['unit'] + ')',
                              basis=lb if rb is None else ('per_' + rb if lb is None else None))
            number(metric['value'])
            metric['source_events'] = list(dict.fromkeys(metric['source_events'] + other['source_events']))
            record = 'D%03d' % (len(self.derived) + 1)
            result = dict(record=record, metric=metric, operation=operation,
                          source=deepcopy(source), right=deepcopy(right))
            self.derived[record] = deepcopy(result)
            self.revision += 1
        if result is None:
            result = super()._calculate(source, operation, right=right, target_unit=None if operation == 'scale' else target_unit,
                                        factor=factor, to_basis=to_basis, factor_note=factor_note if operation == 'scale' else None)
        if operation == 'scale' and target_unit is None and not metric['unit'].startswith('USD'):
            result['metric']['verification_eligible'] = False
            result['unit_status'] = 'unconfirmed'
            result['warning'] = 'Scalar multiplication retains the old unit; intended physical output unit was not stated. This record cannot be used as verified constraint evidence. For a unit change, recalculate from the original T/D source using convert and target_unit.'
        result['annotation'] = factor_note
        result['requested_operation'] = operation
        self.derived[result['record']] = deepcopy(result)
        return result
