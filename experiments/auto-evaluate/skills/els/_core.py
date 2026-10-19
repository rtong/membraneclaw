"""Model-led ELS: native physics, two helpers, non-blocking delivery audit.

No task parser, candidate search, answer lookup or automatic physical execution.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import re

from auto_evaluate.engineering_skills.primitives import ReviewSession, SkillError, convert, number


def obj(properties, required=()):
    return dict(type='object', properties=properties, required=list(required), additionalProperties=False)


def validate(value, schema, path='arguments'):
    kind = schema.get('type')
    valid = {'object': isinstance(value, dict), 'array': isinstance(value, list),
             'string': isinstance(value, str), 'number': type(value) in (int, float),
             'boolean': type(value) is bool}
    if kind and not valid.get(kind, False):
        raise SkillError(path + ': expected ' + kind)
    if kind == 'number':
        number(value)
        if value < schema.get('minimum', value) or value > schema.get('maximum', value):
            raise SkillError(path + ': outside native parameter range')
    if 'enum' in schema and value not in schema['enum']:
        raise SkillError(path + ': use ' + str(schema['enum']))
    if kind == 'object':
        missing = set(schema.get('required', [])) - set(value)
        if missing:
            raise SkillError(path + ': missing ' + str(sorted(missing)))
        for key, item in value.items():
            rule = schema.get('properties', {}).get(key, schema.get('additionalProperties', True))
            if rule is False:
                raise SkillError(path + ': unknown field ' + key)
            if isinstance(rule, dict):
                validate(item, rule, path + '.' + key)
    if kind == 'array':
        if len(value) < schema.get('minItems', 0):
            raise SkillError(path + ': empty list')
        for item in value:
            validate(item, schema['items'], path + '[]')


def unit_basis(unit, basis=None):
    aliases = {'m³/s': 'm3/s', 'm³/h': 'm3/h', 'm³/d': 'm3/day',
               'm3/d': 'm3/day', 'm³/day': 'm3/day', 'kWh/m³': 'kWh/m3',
               'USD/m³': 'USD/m3', 'bar(a)': 'bar'}
    unit = aliases.get(unit, unit)
    match = re.fullmatch(r'(USD(?:/m3)?)[_ -](\d{4})', unit)
    if match:
        inferred = 'USD_' + match[2]
        if basis and basis != inferred:
            raise SkillError('Conflicting currency bases')
        unit, basis = match[1], inferred
    if basis and re.fullmatch(r'\d{4}', basis):
        basis = 'USD_' + basis
    return unit, basis


class ELS:
    protocol_version = 'evidence-linked-skills@0.7.0'

    def __init__(self, question, specs, native_schemas, executor=None, max_calls=20):
        if not specs or type(max_calls) is not int or max_calls < 1:
            raise SkillError('Physical adapters and positive call budget required')
        self.question, self.specs, self.executor = question, deepcopy(specs), executor
        self.audit, self.derived, self.checks, self.drafts = [], {}, [], []
        self.revision, self.reminded, self.submission = 0, False, None
        self.session = ReviewSession(dict(constraints=[], max_calls=max_calls,
                                         execution_origin='ELS_0.7_native_executor'), specs, self._physics)
        native = [deepcopy(s) for s in native_schemas if s['function']['name'] in specs]
        if {s['function']['name'] for s in native} != set(specs):
            raise SkillError('Native schema and adapter names differ')
        ref = obj({'record': {'type': 'string'}, 'metric': {'type': 'string'}}, ['record', 'metric'])
        numeric, string = {'type': 'number'}, {'type': 'string'}
        rule = obj({'metric': string, 'operator': {'type': 'string', 'enum': ['<=', '>=', '<', '>', '==']},
                    'value': numeric, 'unit': string, 'basis': string}, ['metric', 'operator', 'value', 'unit'])
        helpers = [
            dict(type='function', function=dict(name='els_calculate',
                description='Read recorded evidence; convert, scale or compare. Returns a new immutable D record. No retyping values.',
                parameters=obj({'source': ref, 'operation': {'type': 'string',
                    'enum': ['convert', 'scale', 'difference', 'ratio', 'relative_difference']},
                    'right': ref, 'target_unit': string, 'factor': numeric, 'to_basis': string,
                    'factor_note': string}, ['source', 'operation']))),
            dict(type='function', function=dict(name='els_check',
                description='Check model-declared limits against saved records; converts compatible units. This does not certify correct or complete task interpretation.',
                parameters=obj({'records': {'type': 'array', 'items': string, 'minItems': 1},
                    'constraints': {'type': 'array', 'items': rule, 'minItems': 1}}, ['records', 'constraints'])))
        ]
        self.tool_schemas = native + helpers

    def _physics(self, tool, arguments):
        if self.executor is None:
            raise SkillError('No physical executor attached')
        output = self.executor(tool, arguments)
        if isinstance(output, str):
            output = json.loads(output)
        if not isinstance(output, dict) or 'content' in output:
            raise SkillError('Expected local tool JSON, not an MCP wrapper')
        if output.get('error') or output.get('isError') or output.get('status') == 'error':
            return dict(status='error', data={}, error=str(output))
        return dict(status='success', data=output)

    def _call(self, name, arguments):
        spec = self.specs[name]
        effective = {k: v['default'] for k, v in spec['properties'].items() if 'default' in v}
        effective.update(arguments)
        # Defaults are used for interface applicability, never injected into the request.
        for rule in spec.get('conditional_rules', []):
            if all(effective.get(k) == v for k, v in rule['when'].items()):
                forbidden = set(rule.get('forbid', [])) & set(arguments)
                if forbidden:
                    raise SkillError('Inapplicable native parameter: ' + ', '.join(sorted(forbidden)))
        record = 'T%03d' % (self.session._calls + 1)
        event = self.session.call_tool(record, name, arguments)
        metrics, missing = {}, []
        if event['result']['status'] == 'success':
            for field, meta in spec['output_fields'].items():
                try:
                    metrics[meta['metric']] = self.session.record_metric(record, meta['metric'], event['event_id'], field)
                except SkillError as exc:
                    missing.append(dict(field=field, error=str(exc)))
        self.revision += 1
        return dict(record=record, status=event['result']['status'], result=event['result'],
                    metrics=metrics, missing_fields=missing,
                    _observable_events=[dict(event_type='tool_interaction', tool_name=name,
                        arguments=deepcopy(arguments), observation=event['result'], status=event['result']['status'],
                        metadata=dict(physical_call=True, source_event_id=event['event_id'], record=record))],
                    **({'error': event['result']['error']} if event['result'].get('error') else {}))

    def _metric(self, reference):
        record, name = reference['record'], reference['metric']
        if record in self.derived:
            if name != 'value':
                raise SkillError('Derived records use metric=value')
            return deepcopy(self.derived[record]['metric'])
        if not any(e.get('event_id') == record for e in self.session._events):
            raise SkillError('Unknown calculation record: ' + record)
        aliases = {}
        for spec in self.specs.values():
            for field, meta in spec['output_fields'].items():
                for alias in [field, field.rsplit('.', 1)[-1], meta['metric']]:
                    if alias in aliases and aliases[alias] != meta['metric']:
                        aliases[alias] = None
                    else:
                        aliases[alias] = meta['metric']
        return self.session._get_metric(record, aliases.get(name) or name)

    def _calculate(self, source, operation, right=None, target_unit=None, factor=None, to_basis=None, factor_note=None):
        allowed = {'convert': {'target_unit'}, 'scale': {'factor', 'to_basis', 'factor_note'},
                   'difference': {'right'}, 'ratio': {'right'}, 'relative_difference': {'right'}}[operation]
        supplied = {k for k, v in dict(right=right, target_unit=target_unit, factor=factor,
                                      to_basis=to_basis, factor_note=factor_note).items() if v is not None}
        if supplied - allowed:
            raise SkillError('Fields incompatible with ' + operation + '; use convert for unit changes')
        metric = self._metric(source)
        if operation == 'convert':
            if not target_unit:
                raise SkillError('convert requires target_unit')
            target, basis = unit_basis(target_unit)
            if basis and basis != metric.get('basis'):
                raise SkillError('Currency-year changes are not unit conversions')
            metric['value'] = convert(metric['value'], metric['unit'], target)
            metric['unit'] = target
        elif operation == 'scale':
            if factor is None:
                raise SkillError('scale requires an explicit factor')
            metric['value'] *= number(factor)
            if to_basis:
                if not metric['unit'].startswith('USD') or not re.fullmatch(r'USD_\d{4}', to_basis):
                    raise SkillError('to_basis requires an explicit USD currency year')
                metric['basis'] = to_basis
            metric['factor_declaration'] = dict(value=factor, note=factor_note, semantic_verified=False)
        else:
            if right is None:
                raise SkillError(operation + ' requires right={record,metric}')
            other = self._metric(right)
            if metric.get('basis') != other.get('basis'):
                raise SkillError('Currency or physical bases differ')
            rv = convert(other['value'], other['unit'], metric['unit'])
            lv = metric['value']
            if operation != 'difference' and rv == 0:
                raise SkillError('Cannot divide by zero evidence')
            metric['value'] = lv - rv if operation == 'difference' else lv / rv if operation == 'ratio' else (lv - rv) / rv
            if operation != 'difference':
                metric['unit'], metric['basis'] = 'fraction', None
            metric['source_events'] = list(dict.fromkeys(metric['source_events'] + other['source_events']))
        number(metric['value'])
        record = 'D%03d' % (len(self.derived) + 1)
        self.derived[record] = dict(record=record, metric=metric, operation=operation,
                                    source=deepcopy(source), right=deepcopy(right))
        self.revision += 1
        return deepcopy(self.derived[record])

    def _check(self, records, constraints):
        results = []
        operators = {'<=': lambda a, b: a <= b, '>=': lambda a, b: a >= b,
                     '<': lambda a, b: a < b, '>': lambda a, b: a > b, '==': lambda a, b: a == b}
        for record in dict.fromkeys(records):
            for rule in constraints:
                item = dict(record=record, requirement=deepcopy(rule), state='unknown',
                            requirement_origin='model_declared', semantic_verified=False)
                try:
                    metric = self._metric(dict(record=record, metric=rule['metric']))
                    unit, basis = unit_basis(rule['unit'], rule.get('basis'))
                    if metric.get('basis') != basis:
                        raise SkillError('Currency/physical basis missing or different; state the basis explicitly')
                    observed = convert(metric['value'], metric['unit'], unit)
                    threshold = number(rule['value'])
                    item.update(value=observed, unit=unit, source_events=metric['source_events'],
                                state='pass' if operators[rule['operator']](observed, threshold) else 'fail',
                                delta=observed - threshold)
                except SkillError as exc:
                    item['reason'] = str(exc)
                results.append(item)
        report = dict(check_id='C%03d' % (len(self.checks) + 1), revision=self.revision,
                      records=list(records), constraints=deepcopy(constraints), results=results,
                      scope='Declared limits only. Does not certify task coverage, thresholds, prose, selection or optimality.')
        self.checks.append(deepcopy(report))
        return report

    def execute_tool(self, name, arguments):
        event = dict(sequence=len(self.audit) + 1, name=name, arguments=deepcopy(arguments),
                     timestamp_utc=datetime.now(timezone.utc).isoformat())
        self.audit.append(event)
        try:
            schema = next((s['function']['parameters'] for s in self.tool_schemas if s['function']['name'] == name), None)
            if schema is None:
                raise SkillError('Use native physics, els_calculate or els_check')
            validate(arguments, schema)
            if name in self.specs:
                result = self._call(name, arguments)
            else:
                result = {'els_calculate': self._calculate, 'els_check': self._check}[name](**arguments)
        except (SkillError, KeyError, TypeError, ValueError) as exc:
            result = dict(error=str(exc), help='Use returned record IDs and native metric names. Unit changes use convert. You may stop with an incomplete answer.')
        event['result'] = deepcopy(result)
        view = deepcopy(result)
        if 'metrics' in view and 'result' in view:
            # Full raw output stays above; avoid repeating mapped values to the model.
            data = view.pop('result').get('data', {})
            for metric in view['metrics'].values():
                owner, parts = data, metric['field'].split('.')
                for part in parts[:-1]:
                    owner = owner.get(part, {})
                owner.pop(parts[-1], None)
            view['other_outputs'] = {k: v for k, v in data.items() if v}
        event['model_feedback'] = deepcopy(view)
        return json.dumps(view, ensure_ascii=False, allow_nan=False)

    def observe_response(self, raw):
        """Keep provider text even when a later budget/transport gate stops the run."""
        choices = raw.get('choices') or []
        message = choices[0].get('message', {}) if choices else {}
        text = message.get('content')
        if isinstance(text, str) and text.strip():
            self.drafts.append(dict(text=text, has_tool_calls=bool(message.get('tool_calls')),
                                    finish_reason=choices[0].get('finish_reason')))

    def delivery_audit(self):
        current = [c for c in self.checks if c['revision'] == self.revision]
        issues = []
        if not any(e.get('result', {}).get('status') == 'success' for e in self.session._events):
            issues.append('No successful physical evidence')
        if not current:
            issues.append('No current els_check report')
        elif any(r['state'] == 'unknown' for r in current[-1]['results']):
            issues.append('Latest check contains unknown evidence or units')
        return dict(protocol_complete=not issues, issues=issues, current_check=current[-1] if current else None,
                    engineering_correctness='not_certified', prose_consistency='not_certified',
                    scope='Protocol evidence only; failed alternatives do not automatically invalidate a model decision.')

    def accept_final(self, text):
        raw = self.drafts[-1]['text'] if self.drafts and not self.drafts[-1]['has_tool_calls'] and self.drafts[-1]['text'].strip() == text else text
        audit = self.delivery_audit()
        self.submission = dict(answer=raw, complete=audit['protocol_complete'], delivery_audit=audit, accepted_as_text=True)
        if audit['issues'] and not self.reminded:
            self.reminded = True
            raise SkillError('; '.join(audit['issues']) + '. Use els_check on available records if appropriate, then write your answer normally. If you stop, your text will still be retained as incomplete. Do not use els_finish.')
        return raw

    def record_termination(self, reason, detail):
        self.audit.append(dict(sequence=len(self.audit) + 1, termination=reason, detail=detail))

    def export_trace(self):
        return deepcopy(dict(protocol=self.protocol_version, audit_events=self.audit,
                             session=self.session.export_trace(), derived_records=self.derived,
                             check_history=self.checks, model_drafts=self.drafts, submission=self.submission,
                             delivery_audit=self.delivery_audit(), final_reminder_used=self.reminded))
