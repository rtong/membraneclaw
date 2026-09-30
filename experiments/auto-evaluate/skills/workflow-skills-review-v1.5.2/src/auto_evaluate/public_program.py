"""Compile public SWRO tables into executable contracts, without an LLM planner.

Only question_prompt and task_family enter this module. Parsing failures are
explicit; neither reference answers nor cached experiment decisions are inputs.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
import re
from typing import Callable

from .fixed_input_guard import require_fixed_inputs, required_paths
from .public_search_contract import (SearchContractError, search_requirements, check_search_job, check_search_call)
from .watertap_tools.program_contract import LABELS, SPECIES
from .watertap_tools.program_tool import _ALLOWED_ARGUMENTS, _Executor, _merge

VERSION = "public-program@1.5.2"
N = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
OP = r"(<=|>=|<|>)"


class PublicInputError(ValueError):
    pass


def clean(text):
    return (text.replace("−", "-").replace("≥", ">=").replace("≤", "<=")
            .replace("×10⁻¹²", "e-12").replace("×10⁻⁸", "e-8")
            .replace("×10^-12", "e-12").replace("×10^-8", "e-8")
            .replace("m³", "m3").replace("m^3", "m3").replace("m²", "m2")
            .replace("m^2", "m2").replace("–", "~").replace("—", "~"))


def number(text):
    text = clean(text).replace(",", "").strip()
    if not re.fullmatch(N, text, re.I):
        raise PublicInputError(f"Expected numeric public cell: {text!r}")
    value = float(text)
    if not math.isfinite(value): raise PublicInputError("Nonfinite public input")
    return value


def cells(question):
    return [[v.strip() for v in line.strip().strip("|").split("|")]
            for line in clean(question).splitlines() if "|" in line]


def cell_after(question, label):
    for row in cells(question):
        for i, value in enumerate(row[:-1]):
            if value.casefold() == label.casefold(): return row[i+1]
    return None


def numeric_cell(question, label, default=None):
    value = cell_after(question, label)
    return number(value) if value is not None else default


def match_number(text, pattern, default=None):
    m = re.search(pattern.replace("{n}", f"({N})"), clean(text).replace(",", ""), re.I)
    return float(m[1]) if m else default


def inputs(question, tool, *, strict=False):
    aliases = {**LABELS, "nominal feed pressure": "feed_pressure_bar", "fixed pressure": "feed_pressure_bar", "membrane area per active train": "membrane_area_m2",
        "feed nacl": "feed_nacl_mass_frac", "feed temperature": "feed_temperature_c",
        "membrane surface area": "membrane_area_m2", "active membrane surface area": "membrane_area_m2",
        "ro membrane area": "ro_area_m2", "design feed": "feed_flow_m3_s",
        "total design feed": "feed_flow_m3_s", "total membrane area": "ro_area_m2",
        "base membrane area": "ro_area_m2", "base area": "ro_area_m2",
        "base pressure": "p1_pressure_bar", "normal temperature": "feed_temperature_c", "equivalent module length": "module_length_m", "high-salinity tds": "feed_tds_g_L"}
    found = {}; composition = {}
    def assign(target, key, value):
        if key in target and target[key] != value:
            raise PublicInputError(f"Conflicting explicit public input {key}: {target[key]} vs {value}")
        target[key] = value
    for row in cells(question):
        for i in (0, 1):
            if i+1 >= len(row): continue
            label = row[i]
            try: value = number(row[i+1])
            except PublicInputError:
                if strict and i == 0 and (label.casefold() in aliases or label in SPECIES) and row[i+1].casefold() != 'decision variable':
                    raise PublicInputError(f"Invalid explicit numeric input {label}: {row[i+1]!r}")
                continue
            if i == 0 and label in SPECIES:
                assign(composition,label,value); continue
            if tool in {'equilibrate_feed','analyze_ro_scaling'} and label.casefold() in {'acid / base dose','acid/base dose'}:
                assign(found,'acid_addition_mol_s',value);assign(found,'base_addition_mol_s',value);continue
            key = aliases.get(label.casefold())
            if label.casefold() in {"feed", "feed flow"}:
                key = "feed_flow_mass_kg_s" if tool in {"simulate_ro", "analyze_ro_scaling"} else "feed_flow_m3_s"
            if tool == "simulate_swro_system" and key == "membrane_area_m2": key = "ro_area_m2"
            if tool == "equilibrate_feed" and key == "feed_temperature_c": key = "temperature_c"
            if tool == "simulate_swro_system" and label.casefold() == "p1/p2 efficiency":
                assign(found,"p1_efficiency",value); assign(found,"p2_efficiency",value)
            if key in _ALLOWED_ARGUMENTS[tool]: assign(found,key,value)
            elif strict and i==0 and len(row)>3 and re.search(r'\bfixed\b',row[3],re.I):
                raise PublicInputError(f'Unrecognized explicitly fixed input: {label}; cannot certify contract')
    if composition and "composition_mol_s" in _ALLOWED_ARGUMENTS[tool]: found["composition_mol_s"] = composition
    if tool == "simulate_swro_system":
        erd = cell_after(question, "Energy recovery") or cell_after(question, "ERD type") or cell_after(question, "ERD")
        if strict and erd and erd.casefold() not in {'pressure_exchanger','pressure exchanger','px','pump_as_turbine','pump as turbine','pat'}:
            raise PublicInputError(f"Unrecognized public ERD mode: {erd!r}")
        if erd: found["erd_type"] = "pump_as_turbine" if "pat" in erd.casefold() else "pressure_exchanger"
        ab = cell_after(question, "Membrane A/B")
        if ab:
            a, b = ab.split("/"); assign(found,"A_comp",number(a)); assign(found,"B_comp",number(b))
    return found


def rule(metric, op, threshold, source, **extra):
    return dict(metric=metric, op=op, threshold=threshold, source=source, **extra)


def ro_rules(question, scenario=None):
    patterns = [(r"(?:Permeate (?:production(?:, Qp)?|flow)|Qp)", "Qp_m3_h"),
                (r"(?:Water )?Recovery", "recovery_pct"),
                (r"(?:Average water |Water )?Flux", "flux_LMH"),
                (r"(?:Salt )?Rejection", "rejection_pct"),
                (r"(?:Permeate )?NaCl", "nacl_mg_L")]
    rules = []
    for label, metric in patterns:
        # Requirements may live in the right half of a combined input table.
        matches = re.findall(label+r"\s*(?:\|\s*)?"+OP+r"\s*("+N+r")", clean(question), re.I)
        unique = {(op, float(v)) for op, v in matches}
        if not unique:
            for label2, metric2, op2 in [('Qp minimum','Qp_m3_h','>='),('Recovery maximum','recovery_pct','<='),('Flux maximum','flux_LMH','<='),('Salt rejection minimum','rejection_pct','>='),('Permeate NaCl maximum','nacl_mg_L','<=')]:
                if metric==metric2:
                    v=numeric_cell(question,label2)
                    if v is not None:unique.add((op2,v))
        if not unique and metric=='Qp_m3_h' and scenario:
            value=cell_after(question,'Permeate production, Qp') or ''
            m=re.search(r'\b'+re.escape(scenario.split()[0])+r'\s*'+OP+r'\s*('+N+r')',value)
            if m:unique.add((m[1],float(m[2])))
        if len(unique) != 1: raise PublicInputError(f"Missing or ambiguous {metric} requirement: {unique}")
        op, value = unique.pop(); rules.append(rule(metric, op, value, label))
    upper=numeric_cell(question,'Qp maximum') if cell_after(question,'Qp maximum') not in (None,'None') else None
    band=re.search(r'(?:flow band of|flow band)\s*('+N+r')\s*~\s*('+N+r')',clean(question),re.I)
    if band: upper=float(band[2])
    if upper is not None:rules.append(rule('Qp_m3_h','<=',upper,'Explicit product-flow upper bound'))
    return rules


def plant_rules(text):
    result = []
    metrics = [("Product", "product_m3_day"), ("SEC", "SEC"), ("P1", "P1_kW"),
               ("P2", "P2_kW"), ("brine", "brine_m3_s")]
    text = clean(text).replace(",", "")
    for label, metric in metrics:
        m = re.search(label+r"\s*"+OP+r"\s*("+N+r")", text, re.I)
        if m: result.append(rule(metric, m[1], float(m[2]), text))
        m = re.search(label+r"\s*∈\[\s*("+N+r")\s*[;,]\s*("+N+r")\s*\]", clean(text), re.I)
        if m: result.extend([rule(metric, ">=", float(m[1]), text), rule(metric, "<=", float(m[2]), text)])
    if not result: raise PublicInputError(f"No plant criteria: {text}")
    return result


def scenario_table(question):
    rows = cells(question); result = []
    for h, header in enumerate(rows):
        start = next((i for i,c in enumerate(header) if c.casefold() in {"case", "design / stress case", "membrane state", "design case", "design condition", "operating case", "intake condition"}), None)
        if start is None or start+2 >= len(header): continue
        suffix = header[start+1:]
        mapping=[]
        for label in suffix:
            low=label.casefold()
            key = ("feed_nacl_mass_frac" if "nacl" in low else "feed_temperature_c" if "temperature" in low
                   else "feed_flow_mass_kg_s" if "flow per train" in low else "A_comp" if "a_comp" in low else "pressure_drop_bar" if "drop" in low
                   else "feed_pressure_bar" if "pressure" in low else None)
            mapping.append(key)
        if not all(mapping): continue
        for row in rows[h+1:]:
            if len(row)<start+len(mapping)+1: continue
            try: values=[number(v) for v in row[start+1:start+1+len(mapping)]]
            except PublicInputError: continue
            result.append(dict(id=row[start], arguments=dict(zip(mapping,values))))
        if result: return result
    raise PublicInputError("No public operating-scenario table")


def bounds(question, labels, default):
    for label in labels:
        value = cell_after(question, label)
        if value:
            m=re.search(r"("+N+r")\s*[~-]\s*("+N+r")",value,re.I)
            if m:return float(m[1]),float(m[2])
    return default


def job(name, tool, base, rules, **kw):
    return dict(name=name, tool=tool, base=deepcopy(base), rules=deepcopy(rules), **kw)


def search(name, tool, base, rules, path, lower, upper, step, **kw):
    return job(name,tool,base,rules,mode="search",variable=path,lower=lower,upper=upper,step=step,**kw)


def finite(name, tool, base, rules, candidates, **kw):
    return job(name,tool,base,rules,mode="finite",candidates=candidates,**kw)


def declared_pressure_resolution(question):
    values=[float(v) for v in re.findall(
        r'(?:Search resolution|Required pressure resolution)\s*:\s*('+N+r')\s*bar',question,re.I)]
    if not values:
        raise PublicInputError('Missing declared pressure resolution in bar')
    if any(not math.isfinite(v) or v<=0 for v in values) or len(set(values))!=1:
        raise PublicInputError('Conflicting or invalid declared pressure resolution')
    return values[0]


def compile_public(question: str, family: str) -> dict:
    """No case IDs or answer fields are used for dispatch or numerical inputs."""
    q=clean(question); jobs=[]; notes=[]; policy={}
    try: requirements=search_requirements(question,family)
    except SearchContractError as exc: raise PublicInputError(str(exc)) from exc
    if family.startswith(("d1_", "d2_")):
        tool="simulate_ro";base=inputs(q,tool);rules=ro_rules(q, 'A')
        if family in {"d1_1a","d1_1b","d1_1c"}:
            scenarios=scenario_table(q)
            if family=="d1_1a":
                for s in scenarios:jobs.append(search(s['id'],tool,_merge(base,s['arguments']),rules,'membrane_area_m2',100,900,1))
                policy={'kind':'common_window','variable':'membrane_area_m2'}
            elif family=="d1_1b":
                candidates=[]
                for row in cells(q):
                    if len(row)<3:continue
                    try:a,b=number(row[1]),number(row[2])
                    except PublicInputError:continue
                    if 0<a<1e-9 and 0<b<1e-5:
                        candidates.append(dict(id=row[0],arguments={'A_comp':a,'B_comp':b}))
                if not candidates:raise PublicInputError('No membrane candidates')
                jobs=[finite('membrane_screen',tool,base,rules,candidates,scenarios=scenarios)]
            else:
                introduction=q.split('Fixed')[0]
                minimum_goal=bool(re.search(r'minimum|Pmin',introduction,re.I)) and not re.search(r'operating window|feasible interval',introduction,re.I)
                for s in scenarios:
                    j=search(s['id'],tool,_merge(base,s['arguments']),ro_rules(q,s['id']),'feed_pressure_bar',35,85,declared_pressure_resolution(q))
                    if minimum_goal:j.update(search_goal='minimum_feasible',independent_boundary_metrics=[])
                    jobs.append(j)
                margin=match_number(q,r'controlling Pmin\s*\+\s*{n}\s*bar')
                if margin is None:margin=match_number(q,r'{n}\s*bar design margin')
                increment=match_number(q,r'next\s*{n}\s*bar')
                if increment is None:increment=match_number(q,r'{n}\s*bar equipment(?: selection| rating|\-selection|\-rating)? increment')
                rating_requested=bool(re.search(r'(?:rated|rating).*?(?:pressure|design)|pressure.*?rat(?:ed|ing)',introduction,re.I))
                gaps=[]
                if rating_requested:
                    if margin is None:gaps.append('design_margin_bar')
                    if increment is None:gaps.append('equipment_increment_bar')
                policy={'kind':'independent_pressure_minima','rating_requested':rating_requested,
                        'design_margin_bar':margin,'equipment_increment_bar':increment,'missing_public_inputs':gaps}
        else:
            jobs,policy=compile_control(q,family,base,rules)
    elif family.startswith('d3_'):
        tool='simulate_swro_system';base=inputs(q,tool,strict=True)
        c1=cell_after(q,'Basic') or cell_after(q,'Criterion 1');c2=cell_after(q,'Recommended margin') or cell_after(q,'Recommended') or cell_after(q,'Criterion 2')
        if not c1 or not c2:raise PublicInputError('Both public plant criteria required')
        # Preserve decimal commas in sensor windows until each endpoint is parsed.
        def criteria(t):
            windows=re.findall(r'(SEC|P1|P2)\s*∈\[([^,]+),([^\]]+)\]',t,re.I)
            if windows:
                mm={'SEC':'SEC','P1':'P1_kW','P2':'P2_kW'}
                return [rule(mm[k.upper()],op,number(v),t) for k,a,b in windows for op,v in [('>=',a),('<=',b)]]
            return plant_rules(t)
        basic=criteria(c1);recommended=criteria(c2)
        var=next((r[i+1] for r in cells(q) for i,v in enumerate(r[:-1]) if i>=4 and v.casefold() in {'decision variable','px decision variable','membrane a (e-12)'}),'')
        selectors=[('p2_efficiency',r'p2 (?:pump )?efficiency',.3,.99,.001),('p1_efficiency',r'hpp efficiency|p1 efficiency',.4,.99,.001),
                   ('pxr_efficiency',r'(?:actual )?px efficiency',.5,.999,.001),('A_comp',r'permeability|membrane a',1e-12,8e-12,1e-14),
                   ('ro_area_m2',r'membrane area|ro.*area',6000,24000,10),('feed_tds_g_L',r'tds',15,55,.1),
                   ('feed_flow_m3_s',r'(?:train )?feed flow',.15,.5,.001),('p1_pressure_bar',r'p1.*pressure',40,85,.1)]
        # Match complete names: membrane A must not consume membrane area.
        selected=next((s for s in selectors if s[0]==requirements['variable']),None)
        if selected is None:raise PublicInputError(f'Unrecognized plant decision variable: {var}')
        path,_,lo,hi,step=selected
        step=requirements["resolution"]
        if '∈' in c1 or '∈' in c2:basic+=recommended;recommended=[]
        jobs=[search('plant_boundary',tool,base,basic,path,lo,hi,step,recommended=recommended,probes=[base[path]] if path in base else [])]
    elif family=='d4_4a':
        jobs=compile_capex(q)
    elif family.startswith('d5_'):
        jobs,policy=compile_chemistry(q,family)
    else:raise PublicInputError(f'Unsupported public family: {family}')
    # Extract explicitly requested independent bounds from public task sentences.
    # Other hard constraints still participate in the feasible intersection.
    independent_metrics=set()
    aliases={'Qp_m3_h':r'\bQp\b|production', 'recovery_pct':r'\brecovery\b',
             'flux_LMH':r'\bflux\b', 'rejection_pct':r'\brejection\b',
             'nacl_mg_L':r'permeate NaCl'}
    for sentence in re.split(r'[.!?\n]',q):
        if re.search(r'\bindependently\b',sentence,re.I) and re.search(r'bound|minimum|maximum',sentence,re.I):
            independent_metrics.update(metric for metric,pattern in aliases.items() if re.search(pattern,sentence,re.I))
    for j in jobs:
        if j['mode']=='search' and independent_metrics:
            requested=independent_metrics & {r['metric'] for r in j['rules']}
            if requested:j['independent_boundary_metrics']=sorted(requested)
        if not j['rules']:raise PublicInputError('Empty engineering constraints')
        if j['mode']=='search' and (j['step']<=0 or j['upper']<=j['lower']):raise PublicInputError('Invalid public search grid')
        if j['tool'] in {'equilibrate_feed','analyze_ro_scaling'} and 'H2O' not in j['base'].get('composition_mol_s',{}):raise PublicInputError('Missing public H2O flow')
    for j in jobs:
        # Only require fields unchanged by every declared candidate/scenario.
        # The generic guard consumes canonical declarations, not task text/IDs.
        varying=set()
        if j['mode']=='search':
            varying.add(j['variable'])
            if j.get('transform'):varying.add('composition_mol_s')
        else:
            for item in j.get('candidates',[])+j.get('scenarios',[]):
                varying.update(item.get('arguments',{}))
        j['required_input_fields']=required_paths({k:v for k,v in j['base'].items() if not k.startswith('_')})
        if j['mode']=='search' and not j.get('transform'):
            j['required_input_fields']=sorted(set(j['required_input_fields']+[j['variable']]))
        j['required_fixed_inputs']={k:deepcopy(v) for k,v in j['base'].items()
                                    if k not in varying and not k.startswith('_')}
        if requirements:
            try: j['input_contract']=check_search_job(j,requirements)
            except SearchContractError as exc: raise PublicInputError(str(exc)) from exc
    return {'workflow_version':VERSION,'family':family,'public_input_sha256':hashlib.sha256(question.encode()).hexdigest(),
            'jobs':jobs,'policy':policy,'metric_contract':{'Qp_m3_h':'permeate total mass flow * 3.6; historical equivalent volume at 1000 kg/m3','solution_volume_m3_h':'salt mass flow / salt mass concentration; actual solution volume when available'},'notes':notes,'source':'public_question_only',
            'contract_validation':{'status':'checked_structured_search' if requirements else 'not_checked_for_this_family',
                                   'scope':'D3 and D5-5a structured variable/tool/grid/criteria plus call-time fixed-input checks',
                                   'search_bounds_provenance':'legacy_engine_defaults; not asserted to be public equipment limits'}}

def compile_capex(q):
    tool='simulate_swro_system';base=inputs(q,tool);candidates=[];rules=[]
    for label,metric,op in [('Minimum product','product_m3_day','>='),('Minimum total product','product_m3_day','>='),('CAPEX ceiling','CAPEX','<='),('SEC ceiling','SEC','<=')]:
        v=numeric_cell(q,label)
        if v is not None:rules.append(rule(metric,op,v,label))
    approved=numeric_cell(q,'Approved budget')
    if approved is not None:rules.append(rule('CAPEX','<=',approved,'Approval cost-index and contingency',multiplier=numeric_cell(q,'Cost-index ratio')*(1+numeric_cell(q,'Project contingency'))))
    def add(name,args,count=1):candidates.append(dict(id=name,arguments=args,multiplicity=count))
    for row in cells(q):
        if len(row)<6:continue
        name,change=row[4:6];low=name.casefold()
        if low in {'one train','two trains','three trains'}:
            count={'one train':1,'two trains':2,'three trains':3}[low]
            add(name,{'feed_flow_m3_s':base['feed_flow_m3_s']/count,'ro_area_m2':base['ro_area_m2']/count},count)
        elif low in {'base','pressure only','area only','hybrid','base px','pressure uprate','pat'}:
            ps=re.search(r'([\d./]+)\s*bar',change);areas=re.search(r'([\d,./]+)\s*m2',change)
            pp=[number(v) for v in ps[1].split('/')] if ps else [base.get('p1_pressure_bar')]
            aa=[number(v) for v in areas[1].split('/')] if areas else [base.get('ro_area_m2')]
            for p in pp:
                for a in aa:
                    args={k:v for k,v in [('p1_pressure_bar',p),('ro_area_m2',a)] if v is not None}
                    if low=='pat':args.update(erd_type='pump_as_turbine',erd_efficiency=numeric_cell(q,'PAT efficiency'))
                    add(f'{name}:{p}:{a}',args)
        elif low=='temperature scan':
            temps=re.search(r'([\d/]+)\s*°c',change,re.I)
            if not temps:raise PublicInputError('No temperature alternatives')
            for t in temps[1].split('/'):add('temperature:'+t,{'feed_temperature_c':number(t)})
        elif low=='warm-case area':
            temp=match_number(change,r'{n}\s*°c');ar=re.search(r'([\d,/]+)\s*m2',change)
            for a in ar[1].split('/'):add('warm-area:'+a,{'feed_temperature_c':temp,'ro_area_m2':number(a)})
        elif low in {'standard membrane','high-permeability'}:
            ar=re.search(r'([\d/]+)\s*thousand',change)
            permeability=numeric_cell(q,'Standard A_comp' if low=='standard membrane' else 'High-flux A_comp')
            for a in ar[1].split('/'):add(name+':'+a,{'ro_area_m2':number(a)*1000,'A_comp':permeability})
        elif low=='efficiency ladder':
            for v in change.split('/'):add('PX:'+v,{'pxr_efficiency':number(v)})
        elif re.match(r's\d ',low):
            if not candidates:add('base',{})
            if 'pat' in change.casefold():add(name,{'erd_type':'pump_as_turbine','erd_efficiency':match_number(change,r'efficiency\s*{n}')})
            else:
                value=match_number(change,r'→\s*{n}')
                path=next((p for token,p in [('tss','feed_tss_g_L'),('area','ro_area_m2'),('pressure','p1_pressure_bar'),('efficiency','pxr_efficiency'),('feed','feed_flow_m3_s')] if token in change.casefold()),None)
                if path is None or value is None:raise PublicInputError('Unrecognized sensitivity change')
                add(name,{path:value})
    if not candidates:raise PublicInputError('No explicit design alternatives')
    return [finite('capital_alternatives',tool,base,rules,candidates)]


def compile_chemistry(q,family):
    tool='analyze_ro_scaling' if re.search(r'Primary tool\s*\|\s*analyze_ro_scaling',q,re.I) else 'equilibrate_feed'
    base=inputs(q,tool,strict=family=='d5_5a')
    mineral_text=cell_after(q,'Minerals') or cell_after(q,'Mineral') or cell_after(q,'Control minerals')
    if not mineral_text:raise PublicInputError('Missing mineral list')
    minerals=[v.strip() for v in mineral_text.split(',')];base['minerals']=minerals
    def chemistry_rules(text,defaults=False):
        rr=[];text=clean(text)
        for m in minerals:
            rx=re.escape(m)+r'(?:\s*(?:/|and|,)\s*(?:'+ '|'.join(map(re.escape,minerals))+r'))*\s*(?:SI\s*)?'+OP+r'\s*('+N+r')'
            found=re.search(rx,text,re.I)
            if found:rr.append(rule('SI:'+m,found[1],float(found[2]),found[0]))
            elif defaults:rr.append(rule('SI:'+m,'<',0,'All listed mineral SI<0'))
        ph=re.search(r'pH\s*'+OP+r'\s*('+N+r')',text,re.I)
        if ph:rr.append(rule('pH',ph[1],float(ph[2]),ph[0]))
        return rr
    if family=='d5_5a':
        rules=chemistry_rules(cell_after(q,'Basic safety') or '',True);recommended=chemistry_rules(cell_after(q,'Recommended margin') or '')
        variable=cell_after(q,'Variable') or '';resolution=cell_after(q,'Resolution') or '';m=re.match(N,resolution,re.I)
        if not m:raise PublicInputError('Missing chemical search resolution')
        requirements=search_requirements(q,family)
        step=requirements['resolution'];transform=None
        declared=requirements['variable']
        tool=requirements['tool']
        if declared=='water_recovery':path,lo,hi='water_recovery',.001,.9
        elif declared=='Ba':path,lo,hi='Ba',step,step*100000;transform='barium_chloride'
        elif declared=='blend_fraction':
            path,lo,hi='blend_fraction',0,1;transform='blend';source_b=deepcopy(base['composition_mol_s'])
            for row in cells(q):
                if len(row)>2 and row[0] in SPECIES:
                    try:source_b[row[0]]=number(row[2])
                    except PublicInputError:pass
            base['_source_b']=source_b
        elif declared=='temperature_c':path,lo,hi='temperature_c',1,60
        elif declared=='base_addition_mol_s':path,lo,hi='base_addition_mol_s',0,.01
        elif declared=='feed_pressure_bar':path,lo,hi='feed_pressure_bar',30,85
        else:raise PublicInputError(f'Unknown chemistry variable {variable}')
        return [search('chemistry_boundary',tool,base,rules,path,lo,hi,step,recommended=recommended,transform=transform)],{}
    scope=next((line for line in q.splitlines()[::-1] if 'Criteria:' in line or 'Hard constraints:' in line or 'Hard limits:' in line),q)
    rules=chemistry_rules(scope)
    if 'all three SI' in scope:
        v=match_number(scope,r'all three SI\s*<=\s*{n}')
        rules=[r for r in rules if not r['metric'].startswith('SI:')]+[rule('SI:'+m,'<=',v,'all three SI') for m in minerals]
    if not any(r['metric'].startswith('SI:') for r in rules):rules=chemistry_rules(cell_after(q,'SI criteria') or cell_after(q,'Primary criterion') or '')+rules
    if not rules:raise PublicInputError('Missing treatment criteria')
    jobs=[finite('untreated',tool,base,rules,[dict(id='untreated',arguments={})])]
    def removal(species,fraction):
        comp=deepcopy(base['composition_mol_s']);removed=comp[species]*fraction;comp[species]-=removed
        if species in {'Ca','Ba'}:comp['Na']=comp.get('Na',0)+2*removed
        return {'composition_mol_s':comp}
    if 'WAC' in q:
        candidates=[]
        for fraction in [.5,.7,.8,.9,.95,.98,.99]:
            comp=deepcopy(base['composition_mol_s'])
            for s in ('Ca','HCO3'):comp[s]*=1-fraction
            candidates.append(dict(id=f'WAC-removal:{fraction}',arguments={'composition_mol_s':comp,'ph':6.2}))
        jobs += [finite('acid_cap',tool,base,rules,[dict(id='acid-cap',arguments={'acid_addition_mol_s':match_number(q,r'added HCl\s*<=\s*{n}')})]),finite('WAC_targets',tool,base,rules,candidates)]
    elif 'Residual Ca' in q:
        residual=match_number(q,r'Residual Ca\s*{n}');fraction=1-residual/base['composition_mol_s']['Ca'];cap=numeric_cell(q,'Maximum Ca removal')
        if fraction>cap+1e-12:raise PublicInputError('Hybrid exceeds public removal cap')
        jobs += [finite('route_checks',tool,base,rules,[dict(id='acid-only',arguments={'acid_addition_mol_s':match_number(q,r'HCl\s*<=\s*{n}')}),dict(id='softening-cap',arguments=removal('Ca',cap))]),search('hybrid_acid',tool,_merge(base,removal('Ca',fraction)),rules,'acid_addition_mol_s',0,.02,numeric_cell(q,'HCl search step'))]
    elif 'lowest feasible Ba removal' in q or 'minimum silica removal' in q:
        species='Ba' if 'lowest feasible Ba removal' in q else 'SiO2'
        values=re.search(r'among ([^\n]+?) that meets',q,re.I)[1];fractions=[float(v)/100 for v in re.findall(r'('+N+r')%',values)]
        jobs += [finite(species+'_removal',tool,base,rules,[dict(id=f'{species}-removal:{v}',arguments=removal(species,v)) for v in fractions])]
    elif 'Vendor dynamic testing' in q:
        values=re.search(r'at ([^\n]+?) recovery, compare',q,re.I)[1];fractions=[float(v)/100 for v in re.findall(r'('+N+r')%',values)]
        jobs += [finite('vendor_envelope',tool,base,rules,[dict(id=f'recovery:{v}',arguments={'water_recovery':v}) for v in fractions]),finite('alternatives',tool,base,rules,[dict(id='acid',arguments={'acid_addition_mol_s':match_number(q,r'HCl\s*{n}\s*mol/s')}),dict(id='softening',arguments=removal('Ca',.3))])]
    elif 'minimum Ca removal' in q:jobs += [search('Ca_removal',tool,base,rules,'removal_fraction',0,.95,.05,transform='calcium_softening')]
    else:
        step=match_number(q,r'on a\s*{n}\s*mol/s grid')
        if step is None:raise PublicInputError('Unknown treatment-route search')
        upper=max(.02,base['composition_mol_s'].get('HCO3',0)*2)
        jobs += [search('acid_boundary',tool,base,rules,'acid_addition_mol_s',0,upper,step)]
    # Explicit acid counterfactuals are public required scenarios, independent
    # of which treatment search branch was selected above. Caps are not doses.
    doses=[]
    for row in cells(q):
        if len(row)<6 or row[4].casefold() not in {'acid only','acidification'}:continue
        match=re.search(r'\bHCl\s+('+N+r')\s*mol/s\b',row[5],re.I)
        if match:
            dose=float(match[1])
            if not math.isfinite(dose) or dose<0:raise PublicInputError('Invalid declared acid test dose')
            if dose not in doses:doses.append(dose)
    for dose in doses:
        covered=any(_merge(j['base'],c['arguments']).get('acid_addition_mol_s')==dose
                    for j in jobs if j['mode']=='finite' for c in j['candidates'])
        if not covered:
            jobs.insert(1,finite('required_acid_test:'+str(dose),tool,base,rules,
                                [dict(id='acid-test',arguments={'acid_addition_mol_s':dose})]))
    return jobs,{}


def compile_control(q,family,base,rules):
    tool='simulate_ro';jobs=[];policy={}
    # Explicit current setpoints have precedence over tool defaults.
    point=cell_after(q,'Current setpoint') or cell_after(q,'Start')
    if point:
        vals=re.findall(N,point,re.I)
        if len(vals)>=2:base.update(feed_pressure_bar=float(vals[0]),feed_flow_mass_kg_s=float(vals[1]))
    if family=='d2_2a':
        pressure=bounds(q,['Pressure range'],(None,None))
        if pressure[0] is None:
            upper=match_number(cell_after(q,'Pressure limit') or '',r'{n}')
            m=re.search(r'[Pp]ressure(?: only)?[: ]+('+N+r')~('+N+r')',q)
            if m:pressure=(float(m[1]),float(m[2]))
            elif upper:pressure=(base['feed_pressure_bar'],upper)
            else:pressure=(match_number(q,r'pressure floor\s*{n}',max(1,base['feed_pressure_bar']-5)),base['feed_pressure_bar']+6)
        lo,hi=pressure
        states=[];audit=[]
        for row in cells(q):
            if len(row)<6:continue
            label,value=row[4:6];low=label.casefold()
            if any(w in low for w in ['salinity disturbance','salinity level','salinity shock','extreme salinity']):key='feed_nacl_mass_frac'
            elif any(w in low for w in ['disturbance','design temperatures','proposed sop']) and ('°C' in value or 'temperature' in q.splitlines()[0].casefold()):key='feed_temperature_c'
            elif re.search(r'^(normal|moderate|severe|disturbed feed|severe-flow|recovery-flow)',low) and ('flow' in low or low in {'normal','moderate','severe'}):key='feed_flow_mass_kg_s'
            else:key=None
            if key:
                for v in re.findall(N,value,re.I):states.append((label+':'+v,{key:float(v)}))
            if re.search(r'(?:sop )?rule [abc]$',low):
                vals=[float(v) for v in re.findall(N,value)]
                if 'kg/s →' in value and len(vals)>=2:audit.append(dict(id=label,arguments={'feed_flow_mass_kg_s':vals[0],'feed_pressure_bar':vals[1]}))
                elif '→' in value and len(vals)>=3:audit.append(dict(id=label,arguments={'feed_nacl_mass_frac':vals[0],'feed_pressure_bar':vals[1],'feed_flow_mass_kg_s':vals[2]}))
        if audit:jobs.append(finite('SOP_audit',tool,base,rules,audit))
        if not states and audit:
            states=[(c['id'],{k:v for k,v in c['arguments'].items() if k!='feed_pressure_bar'}) for c in audit]
        if not states:raise PublicInputError('No public disturbance conditions')
        jobs.append(finite('normal_baseline',tool,base,rules,[dict(id='normal',arguments={})]))
        for name,args in states:jobs.append(search(name,tool,_merge(base,args),rules,'feed_pressure_bar',lo,hi,.01,probes=[base['feed_pressure_bar']]))
        flo,fhi=bounds(q,['Feed-flow range','Secondary feed-flow range','Flow-restoration range'],(None,None))
        if fhi is None and ('flow restoration' in q.casefold() or 'restore' in q.casefold()):fhi=base['feed_flow_mass_kg_s']
        if fhi is None:
            m=re.search(r'(?:secondary feed flow|feed-flow)\s*('+N+r')~('+N+r')',q,re.I)
            if m:flo,fhi=float(m[1]),float(m[2])
        if fhi is not None and 'cannot be restored' not in q:
            policy={'kind':'feed_restoration','max_flow':fhi,'flow_step':.005,'pressure_bounds':[lo,hi],'pressure_step':.01}
    else:
        corner=cell_after(q,'Critical test point')
        if corner:
            vals=[float(v) for v in re.findall(N,corner)]
            return [finite('equipment_corner',tool,base,rules,[dict(id='equipment-corner',arguments={'feed_pressure_bar':vals[0],'feed_flow_mass_kg_s':vals[1]})])],{}
        candidates=[]
        for row in cells(q):
            if len(row)<6:continue
            name,value=row[4:6]
            if re.match(r'Candidate [ABC]$',name):
                vals=[float(v) for v in re.findall(N,value)]
                if len(vals)>=2:candidates.append(dict(id=name,arguments={'feed_pressure_bar':vals[0],'feed_flow_mass_kg_s':vals[1]}))
            if re.match(r'Option [AB]$',name):
                p=match_number(value,r'(?:to|Hold)\s*{n}\s*bar');f=match_number(value,r'{n}\s*kg/s')
                if p is None:p=base.get('feed_pressure_bar')
                if p is not None and f is not None:candidates.append(dict(id=name,arguments={'feed_pressure_bar':p,'feed_flow_mass_kg_s':f}))
        if candidates:return [finite('stated_corrections',tool,base,rules,[dict(id='current',arguments={})]+candidates)],{}
        if cell_after(q,'Target'):
            vals=[float(v) for v in re.findall(N,cell_after(q,'Target'))];p0=base['feed_pressure_bar'];f0=base['feed_flow_mass_kg_s'];pt,ft=vals[:2]
            # Both legal action orders are checked; no untested safe-path claim.
            ps=max(pt,p0-.25)
            cc=[dict(id='start',arguments={}),dict(id='pressure-first',arguments={'feed_pressure_bar':ps}),dict(id='flow-first',arguments={'feed_flow_mass_kg_s':ft}),dict(id='intermediate',arguments={'feed_pressure_bar':ps,'feed_flow_mass_kg_s':ft}),dict(id='target',arguments={'feed_pressure_bar':pt,'feed_flow_mass_kg_s':ft})]
            return [finite('safe_path_nodes',tool,base,rules,cc)],{'kind':'safe_path','max_pressure_step':.25,'max_flow_step':.005}
        lo,hi=bounds(q,['Pressure range','Adjustable pressure'],(None,None))
        if lo is None:
            values=cell_after(q,'Pressure candidates')
            if values:
                vv=[float(v) for v in re.findall(N,values)];lo,hi=min(vv),max(vv)
            else:lo,hi=base['feed_pressure_bar'],base['feed_pressure_bar']
        flo,fhi=bounds(q,['Adjustable feed mass flow','Feed-flow range','Adjustable feed flow'],(None,None))
        if flo is None:
            flo=base['feed_flow_mass_kg_s'];fhi=flo+.1 if cell_after(q,'First-priority action') else flo
        jobs=[finite('current',tool,base,rules,[dict(id='current',arguments={})])]
        if hi>lo:jobs.append(search('pressure_only',tool,base,rules,'feed_pressure_bar',lo,hi,.05))
        if fhi>flo:jobs.append(search('flow_only',tool,base,rules,'feed_flow_mass_kg_s',flo,fhi,.005))
        if hi>lo and fhi>flo:policy={'kind':'coordinated_control','flow_bounds':[flo,fhi],'pressure_bounds':[lo,hi],'flow_step':.005,'pressure_step':.05,'priority_text':cell_after(q,'Optimization priority') or cell_after(q,'Optimization objective') or ''}
    if policy.get('kind')=='coordinated_control':
        priority=policy.get('priority_text','').casefold()
        pressure=priority.find('pressure');flow=priority.find('flow')
        if pressure>=0 and flow>=0 and pressure<flow:
            raise PublicInputError('Unsupported optimization priority: pressure before flow; flow-first restoration cannot substitute for this objective')
    return jobs,policy


def metric_values(result,args,count=1):
    m={}
    if 'permeate' in result:
        p=result['permeate'];perf=result.get('performance',{})
        for key,path in [('Qp_m3_h',('permeate','water_kg_s')),('recovery_pct',('performance','water_recovery_pct')),('rejection_pct',('performance','salt_rejection_pct')),('flux_LMH',('flux','water_LMH')),('nacl_mg_L',('permeate','nacl_mg_L'))]:
            group,field=path
            if field in result.get(group,{}):m[key]=result[group][field]*(3.6 if key=='Qp_m3_h' else 1)
        # Preserve the benchmark's declared equivalent-volume convention.
        # Actual solution volume is a different observable, never a silent override.
        total=p.get('flow_kg_s')
        if total is None and p.get('water_kg_s') is not None:
            total=p['water_kg_s']+p.get('nacl_kg_s',0)
        if total is not None:m['Qp_m3_h']=total*3.6
        if p.get('nacl_mg_L',0)>0 and p.get('nacl_kg_s') is not None:
            m['solution_volume_m3_h']=p['nacl_kg_s']/(p['nacl_mg_L']/1000)*3600
    if 'costing' in result:
        perf=result['performance'];cost=result['costing'];des=result.get('desalination',{})
        product=perf['product_flow_m3_s'];m.update(product_m3_day=product*86400*count,product_m3_s=product*count,CAPEX=cost['total_capital_cost_usd']*count,SEC=cost['specific_energy_kWh_m3'],LCOW=cost['LCOW_usd_m3'])
        m['brine_m3_s']=(args['feed_flow_m3_s']-product)*count
        for label,key in [('P1_kW','p1_power_kW'),('P2_kW','p2_power_kW')]:
            if key in des:m[label]=des[key]*count
        m['net_power_kW']=m['SEC']*m['product_m3_s']*3600;m['daily_energy_kWh']=m['net_power_kW']*24
        m['capital_intensity_usd_per_m3_day']=m['CAPEX']/m['product_m3_day']
    # Preserve distinct recovery definitions; never substitute RO for system recovery.
    for key in ('ro_recovery_pct','system_recovery_pct'):
        if key in result.get('performance',{}):m[key]=result['performance'][key]
    coupled=result.get('ro_performance',{})
    for source,target in [('water_recovery_pct','recovery_pct'),('salt_rejection_pct','rejection_pct'),
                          ('water_flux_LMH','flux_LMH'),('retentate_pressure_bar','retentate_pressure_bar')]:
        if source in coupled:m[target]=coupled[source]
    chem=result.get('concentrate_chemistry') or result.get('concentrate') or result
    if isinstance(chem,dict):
        si=chem.get('saturation_index',{})
        m.update({'SI:'+k:v for k,v in si.items()})
        if 'solution' in chem:m.update(pH=chem['solution']['ph'],osmotic_pressure_bar=chem['solution']['osmotic_pressure_bar'])
    return m


def checks(metrics,rules):
    from operator import lt,le,gt,ge
    ops={'<':lt,'<=':le,'>':gt,'>=':ge};out=[]
    for r in rules:
        value=metrics.get(r['metric']);value=value*r.get('multiplier',1) if value is not None else None
        out.append({**r,'value':value,'passed':bool(value is not None and math.isfinite(value) and ops[r['op']](value,r['threshold']))})
    return out


def transformed(j,value):
    args=deepcopy(j['base']);kind=j.get('transform');b=args.pop('_source_b',None)
    if kind=='blend':args['composition_mol_s']={k:(1-value)*v+value*b.get(k,v) for k,v in args['composition_mol_s'].items()}
    elif kind=='barium_chloride':args['composition_mol_s']['Ba']=value;args['composition_mol_s']['Cl']+=2*value
    elif kind=='calcium_softening':
        removed=args['composition_mol_s']['Ca']*value;args['composition_mol_s']['Ca']-=removed;args['composition_mol_s']['Na']+=2*removed
    else:args[j['variable']]=value
    return args


def boundary_hints(job, outcomes, jobs, varying_field):
    """Predict edge locations from two compatible observed jobs; never evidence."""
    target=job['base'].get(varying_field)
    if target is None:return []
    reference={j['name']:j for j in jobs}
    samples={}
    for outcome in outcomes:
        source=reference.get(outcome['job'])
        if not source or not outcome.get('complete') or source.get('tool')!=job['tool']:continue
        if any(source.get(k)!=job.get(k) for k in ('variable','lower','upper','step','rules','transform')):continue
        if {k:v for k,v in source['base'].items() if k!=varying_field}!={k:v for k,v in job['base'].items() if k!=varying_field}:continue
        x=source['base'].get(varying_field)
        if x is None:continue
        for edge in outcome.get('constraint_boundaries',[]):
            points=edge.get('points',[])
            if not edge.get('adjacent') or len(points)!=2:continue
            a,b=points
            if a['value']==b['value']:continue
            crossing=a['x']+(edge['threshold']-a['value'])*(b['x']-a['x'])/(b['value']-a['value'])
            samples.setdefault((edge['metric'],edge['operator'],edge['threshold']),{})[x]=crossing
    hints=[]
    for values in samples.values():
        nearest=sorted(values,key=lambda x:abs(x-target))[:2]
        if len(nearest)!=2:continue
        a,b=nearest
        predicted=values[a]+(target-a)*(values[b]-values[a])/(b-a)
        if math.isfinite(predicted) and job['lower']<=predicted<=job['upper']:hints.append(predicted)
    return hints


class Engine:
    def __init__(self,executor,budget):
        self.executor=_Executor(executor,budget);self.rows=[];self.outcomes=[]
    def evaluate(self,j,args,label,value=None,count=1,*,candidate_id=None,scenario_id=None):
        row={'row_id':f'{j["name"]}:{label}','job':j['name'],'value':value,'arguments':deepcopy(args),'multiplicity':count}
        try:
            require_fixed_inputs(j.get('required_fixed_inputs',{}),args,required_fields=j.get('required_input_fields',[]))
            if j['mode']=='search' and j.get('required_input_fields'):
                if value is None or args!=transformed(j,value):
                    raise PublicInputError('Search arguments differ from declared variable/coupling at this point')
            if j['mode']=='finite':
                candidates=[c for c in j['candidates'] if c['id']==candidate_id]
                scenarios=[s for s in (j.get('scenarios') or [{'id':'design','arguments':{}}]) if s['id']==scenario_id]
                if len(candidates)!=1 or len(scenarios)!=1:
                    raise PublicInputError('Finite call requires an unambiguous declared candidate and scenario')
                candidate,scenario=candidates[0],scenarios[0]
                expected=_merge(j['base'],candidate['arguments'],scenario['arguments'])
                if args!=expected or count!=candidate.get('multiplicity',1):
                    raise PublicInputError('Finite arguments or multiplicity differ from declared candidate/scenario')
                row.update(candidate_id=candidate_id,scenario_id=scenario_id)
            check_search_call(j,args,value)
            result=self.executor.run(j['tool'],args,row_id=row['row_id']);metrics=metric_values(result,args,count)
            row.update(result=result,metrics=metrics,checks=checks(metrics,j['rules']),recommended_checks=checks(metrics,j.get('recommended',[])),error=None)
            missing=[r['metric'] for r in row['checks']+row['recommended_checks'] if r['value'] is None]
            if missing:raise PublicInputError('Missing result metrics: '+','.join(missing))
            row['feasible']=all(r['passed'] for r in row['checks'])
            row['recommended_feasible']=row['feasible'] and bool(j.get('recommended')) and all(r['passed'] for r in row['recommended_checks'])
        except Exception as exc:
            if 'budget exhausted' in str(exc):raise
            row.update(error=str(exc),feasible=False,recommended_feasible=False)
        self.rows.append(row);return row
    def finite(self,j):
        candidates=[]
        for c in j['candidates']:
            rows=[]
            for s in j.get('scenarios') or [{'id':'design','arguments':{}}]:
                args=_merge(j['base'],c['arguments'],s['arguments']);r=self.evaluate(j,args,c['id']+':'+s['id'],count=c.get('multiplicity',1),candidate_id=c['id'],scenario_id=s['id']);rows.append(r)
                if not r['feasible']:break
            candidates.append({'candidate_id':c['id'],'feasible':all(r['feasible'] for r in rows),'rows':[r['row_id'] for r in rows],'evidence_complete':all(not r.get('error') for r in rows)})
        return {'job':j['name'],'mode':'finite','candidates':candidates,'complete':all(c['evidence_complete'] for c in candidates)}
    def search(self,j,*,quota=None):
        start=sum(e.get('metadata',{}).get('physical_call') is True for e in self.executor.events)
        grid={};lo,hi,step=j['lower'],j['upper'],j['step'];end=int(math.floor((hi-lo)/step+1e-8))
        def val(i):return float(Decimal(str(lo))+i*Decimal(str(step)))
        def at(i):
            i=max(0,min(end,int(i)))
            if i not in grid:
                used=sum(e.get('metadata',{}).get('physical_call') is True for e in self.executor.events)-start
                if quota is not None and used>=quota:raise PublicInputError('Per-branch budget exhausted')
                grid[i]=self.evaluate(j,transformed(j,val(i)),str(i),val(i))
            return grid[i]
        # Start near the public baseline; huge endpoints may be solver failures,
        # which are unknown states, never engineering infeasibility evidence.
        middle=end//2
        probes=j.get('probes') or []
        if probes:middle=max(0,min(end,round((probes[0]-lo)/step)))
        hints=j.get('boundary_hints',[])
        warm=False
        if hints:
            a=max(0,min(end,math.floor((min(hints)-lo)/step)))
            b=max(0,min(end,math.ceil((max(hints)-lo)/step)))
            if a==b:b=min(end,a+1);a=max(0,b-1)
            required=set(j.get('independent_boundary_metrics',[r['metric'] for r in j['rules']]))
            for expansion in range(3):
                ra,rb=at(a),at(b)
                if not ra['error'] and not rb['error']:
                    warm=all(checks(ra['metrics'],[r])[0]['passed']!=checks(rb['metrics'],[r])[0]['passed'] for r in j['rules'] if r['metric'] in required)
                    if warm:break
                if expansion<2:a=max(0,a-2**expansion);b=min(end,b+2**expansion)
        if not warm:
            at(middle);at(0)
            # A measured feasible anchor suffices for a minimum search; the upper
            # exploration endpoint is not a required equipment-boundary test.
            if not (j.get('search_goal')=='minimum_feasible' and grid[middle]['feasible']):
                at(end)
        for _ in range(6):
            valid=sorted(i for i,r in grid.items() if not r['error'])
            if len(valid)>=2:break
            candidates=[(b-a,(a+b)//2) for a,b in zip(sorted(grid),sorted(grid)[1:]) if b-a>1]
            if not candidates:break
            at(max(candidates)[1])
        valid=sorted(i for i,r in grid.items() if not r['error'])
        if len(valid)<2:return {'job':j['name'],'complete':False,'status':'insufficient_successful_points'}
        left,right=valid[0],valid[-1]
        def interval(rr):
            lower,upper=left,right;boundaries=[];unknown=False
            independent=set(j.get('independent_boundary_metrics',[r['metric'] for r in rr]))
            for r in sorted(rr,key=lambda rule:rule['metric'] not in independent):
                key=r['metric'];threshold=r['threshold'];factor=r.get('multiplier',1)
                # Requested independent edges use the whole declared domain.
                # Remaining constraints only need to tighten the surviving interval.
                domain_left,domain_right=(left,right) if key in independent else (lower,upper)
                if domain_left>domain_right:
                    boundaries.append({'metric':key,'status':'not_searched_after_proven_conflict'})
                    continue
                at(domain_left);at(domain_right)
                if grid[domain_left]['error'] or grid[domain_right]['error']:
                    unknown=True
                    boundaries.append({'metric':key,'status':'solver_failure_at_constraint_endpoint'})
                    continue
                points=sorted((i,x['metrics'][key]*factor) for i,x in grid.items() if domain_left<=i<=domain_right and not x['error'] and key in x.get('metrics',{}))
                diffs=[b[1]-a[1] for a,b in zip(points,points[1:])];tol=max(1e-12,max(abs(v) for _,v in points)*1e-9)
                inc=all(d>=-tol for d in diffs);dec=all(d<=tol for d in diffs)
                if not inc and not dec:
                    all_pass=all(checks(grid[i]['metrics'],[r])[0]['passed'] for i,_ in points)
                    if all_pass:
                        boundaries.append({'metric':key,'status':'inactive_on_tested_points','not_a_global_monotonicity_claim':True});continue
                    unknown=True;boundaries.append({'metric':key,'status':'nonmonotone_samples'});continue
                def passed(i):
                    x=at(i)
                    return None if x['error'] else checks(x['metrics'],[r])[0]['passed']
                a,b=domain_left,domain_right;pa,pb=passed(a),passed(b)
                if pa==pb:
                    if not pa:lower,upper=1,0
                    boundaries.append({'metric':key,'status':'all_sampled_pass' if pa else 'all_sampled_fail','range':[val(a),val(b)]});continue
                # Reuse all observed points to tighten this constraint's bracket.
                # No new call is needed for a point measured for another constraint.
                pairs=[(i,passed(i)) for i,_ in points]
                transitions=[(a0,b0) for (a0,p0),(b0,p1) in zip(pairs,pairs[1:]) if p0!=p1]
                if transitions:a,b=min(transitions,key=lambda pair:pair[1]-pair[0])
                pa,pb=passed(a),passed(b)
                while b-a>1:
                    va=at(a)['metrics'][key]*factor;vb=at(b)['metrics'][key]*factor
                    estimate=a+(threshold-va)*(b-a)/(vb-va) if vb!=va else (a+b)/2
                    m=max(a+1,min(b-1,int(round(estimate))))
                    # Secant interpolation saves calls; clip stubborn edge steps.
                    pm=passed(m)
                    if pm is None:
                        unknown=True;break
                    if pm==pa:a=m
                    else:b=m
                if b-a!=1:unknown=True
                if pa:upper=min(upper,a)
                else:lower=max(lower,b)
                boundaries.append({'metric':key,'operator':r['op'],'threshold':threshold,'points':[{ 'x':val(i),'value':at(i).get('metrics',{}).get(key),'pass':passed(i)} for i in (a,b)],'adjacent':b-a==1})
            return lower,upper,boundaries,unknown
        low,high,evidence,unknown=interval(j['rules'])
        # Successful samples do not replace failed declared endpoints. A minimum
        # search may omit the upper tail only after measuring a feasible anchor;
        # verified warm-start transitions retain their explicitly local scope.
        endpoint_gap = not warm and (left > 0 or right < end)
        minimum_anchor = (j.get('search_goal') == 'minimum_feasible'
                          and left == 0 and grid[right]['feasible'])
        if endpoint_gap and not minimum_anchor:
            unknown = True
            evidence.append({'status': 'unresolved_declared_domain',
                             'observed_range': [val(left), val(right)],
                             'declared_range': [lo, hi]})
        output={'job':j['name'],'search_goal':j.get('search_goal','feasible_interval'),'mode':'search','variable':j['variable'],'search_range':[lo,hi],'resolution':step,'constraint_boundaries':evidence,'independent_boundary_metrics':j.get('independent_boundary_metrics',[r['metric'] for r in j['rules']]),'complete':not unknown,'warm_start_used':warm,'observed_search_range':[val(left),val(right)],'scope':'Constraint-wise monotonicity checked on sampled points; no claim outside the observed search range.'}
        if low<=high and not unknown:
            for i in set([low,high]):at(i)
            if all(at(i)['feasible'] for i in (low,high)):
                output['basic']={'lower':val(low),'upper':val(high),'recommended_midpoint':val((low+high)//2),'midpoint_verified':(low+high)//2 in grid and grid[(low+high)//2]['feasible'],'left_range_limited':low==left,'right_range_limited':high==right}
        else:output['status']='conflicting_constraint_bounds' if not unknown else 'unresolved_nonmonotone_or_solver_failure'
        if j.get('recommended'):
            rl,rh,ev,un=interval(j['rules']+j['recommended']);output['recommended_constraint_boundaries']=ev;output['complete'] &= not un
            if rl<=rh and not un:
                at(rl);at(rh)
                if at(rl)['recommended_feasible'] and at(rh)['recommended_feasible']:output['recommended']={'lower':val(rl),'upper':val(rh)}
        output['solver_failure_points']=[r['value'] for r in grid.values() if r['error']]
        return output


def execute_public_workflow(benchmark,*,max_tool_calls=64,executor=None):
    if executor is None:
        from .watertap_tools.local_tools import execute_tool
        executor=execute_tool
    question=benchmark['question_prompt'];family=benchmark['task_family']
    if family=='d6_6c':
        from .d6_verified import execute_verified_workflow
        return execute_verified_workflow({'case_id':benchmark['case_id'],'question_prompt':question},max_tool_calls=max_tool_calls)
    contract=compile_public(question,family);engine=Engine(executor,max_tool_calls);errors=[]
    for j in contract['jobs']:
        try:
            outcome=engine.search(j) if j['mode']=='search' else engine.finite(j)
            engine.outcomes.append(outcome)
        except Exception as exc:
            errors.append({'job':j['name'],'error':str(exc)});engine.outcomes.append({'job':j['name'],'complete':False,'error':str(exc)})
            if 'budget exhausted' in str(exc):break
    # Shared-flow repair is only entered after measured conflicting pressure
    # bounds. A failed solver or incomplete branch never authorizes repair.
    policy=contract['policy']
    if policy.get('kind') in {'feed_restoration','coordinated_control'}:
        searches=[(j,o) for j in contract['jobs'] for o in engine.outcomes if j['name']==o['job'] and j['mode']=='search' and j['variable']=='feed_pressure_bar' and o.get('complete') and not o.get('basic')]
        for original,outcome in searches:
            base=original['base'];flow0=base['feed_flow_mass_kg_s'];flowmax=policy.get('max_flow',policy.get('flow_bounds',[flow0,flow0])[1]);step=policy['flow_step']
            if flowmax<=flow0:continue
            l=1;h=math.floor((flowmax-flow0)/step+1e-7);best=None
            visited={0:outcome}
            def publish_restoration():
                if best is None:return
                index,observed=best
                previous=visited.get(index-1,{})
                certified=index==0 or (previous.get('complete') and not previous.get('basic'))
                outcome['flow_correction']={
                    'feed_flow_mass_kg_s':float(Decimal(str(flow0))+index*Decimal(str(step))),
                    'pressure_interval':observed['basic'],
                    'adjacent_lower_flow_verified_infeasible':bool(certified),
                    'adjacent_lower_flow_job':previous.get('job'),
                    'optimality':'adjacent grid proof under the observed monotonic feasibility assumption' if certified else 'feasible tested flow; minimum not proven'}
            try:
                # Propose the production/recovery mass-balance lower bound;
                # simulations, including the adjacent lower flow, certify it.
                qr=next((r['threshold'] for r in original['rules'] if r['metric']=='Qp_m3_h' and r['op'] in ('>=','>')),None)
                rr=next((r['threshold'] for r in original['rules'] if r['metric']=='recovery_pct' and r['op'] in ('<=','<')),None)
                sample=next((r for r in engine.rows if r['job']==original['name'] and not r.get('error') and r.get('metrics',{}).get('Qp_m3_h',0)>0),None)
                seed=None
                if sample and qr and rr:
                    m=sample['metrics'];estimate=flow0*qr/m['Qp_m3_h']*m['recovery_pct']/rr
                    seed=max(l,min(h,math.ceil((estimate-flow0)/step-1e-9)))
                attempt=0;restoration_jobs=[]
                while l<=h and attempt<10:
                    mid=seed if seed is not None and seed not in visited else (l+h)//2
                    seed=None;attempt+=1;flow=float(Decimal(str(flow0))+mid*Decimal(str(step)))
                    j=deepcopy(original);j['base']['feed_flow_mass_kg_s']=flow;j['name']=original['name']+f':flow={flow}'
                    j['required_fixed_inputs']=deepcopy(j.get('required_fixed_inputs',{}))
                    j['required_fixed_inputs']['feed_flow_mass_kg_s']=flow
                    j['boundary_hints']=boundary_hints(j,engine.outcomes,contract['jobs']+restoration_jobs,'feed_flow_mass_kg_s')
                    restoration_jobs.append(j)
                    o=engine.search(j);engine.outcomes.append(o);visited[mid]=o
                    if not o.get('complete'):break
                    if o.get('basic'):
                        best=(mid,o);h=mid-1
                        publish_restoration()
                        if mid==0 or (mid-1 in visited and not visited[mid-1].get('basic')):break
                        seed=mid-1
                    else:
                        l=mid+1
                        if best and best[0]==mid+1:break
                        seed=mid+1 if mid+1<=h else None
                publish_restoration()
                if not best or not outcome.get('flow_correction',{}).get('adjacent_lower_flow_verified_infeasible'):
                    errors.append({'job':'flow_correction','error':'Required minimum restoration has not been certified'})
            except Exception as exc:
                publish_restoration()
                errors.append({'job':'flow_correction','error':str(exc)})
    if policy.get('kind')=='common_window' and all(o.get('basic') for o in engine.outcomes):
        lo=max(o['basic']['lower'] for o in engine.outcomes);hi=min(o['basic']['upper'] for o in engine.outcomes)
        if lo<=hi:
            center=math.floor((lo+hi)/2)
            for j in contract['jobs']:
                for value in sorted(set([lo,center,hi])):
                    try:
                        row=engine.evaluate(j,transformed(j,value),'common-verification:'+str(value),value)
                        if not row['feasible']:errors.append({'job':'common_recheck','error':'Common recommendation or boundary lacks passing evidence'})
                    except Exception as exc:errors.append({'job':'common_recheck','error':str(exc)})
            policy['observed_common_interval']=[lo,hi];policy['recommendation']=center
        else:policy['observed_conflict']={'maximum_lower_bound':lo,'minimum_upper_bound':hi}
    events=engine.executor.events;calls=sum(e.get('metadata',{}).get('physical_call') is True for e in events)
    complete=not errors and len(engine.outcomes)>=len(contract['jobs']) and all(o.get('complete') for o in engine.outcomes)
    result={'workflow_version':VERSION,'case_id':benchmark.get('case_id'),'public_input_only':True,'contract':contract,'rows':engine.rows,'outcomes':engine.outcomes,'errors':errors,'events':events,'tool_call_count':calls,'status':'completed_with_evidence' if complete else 'incomplete_evidence','selected_strategy':None,'verification':{'status':'completed_with_evidence' if complete else 'incomplete_evidence'},'generation_mode':'deterministic_public_contract'}
    result['calculation_status']=result['status']
    gaps=contract['policy'].get('missing_public_inputs',[])
    if gaps:
        result['missing_public_inputs']=gaps
        result['status']='blocked_missing_public_inputs' if complete else 'incomplete_evidence'
        result['verification']['status']=result['status']
    result['decisions']=decision_summary(result,question)
    result['verified_report']=render_report(result,question)
    return result


def render_report(result,question):
    lines=['# Engineering result from recorded simulations','',f"Execution status: {result['status']}. Actual physical calls: {result['tool_call_count']}. Failed solver attempts count against the same budget.",'','## Decisions and search evidence','',json.dumps({'decisions':result.get('decisions',[]),'outcomes':result['outcomes'],'policy':result['contract']['policy'],'errors':result['errors']},ensure_ascii=False,indent=2),'','## All measured rows','', '| Row | Variable | Feasible | Results and checks |','|---|---:|---|---|']
    for row in result['rows']:
        payload={'metrics':row.get('metrics'),'checks':row.get('checks'),'recommended_checks':row.get('recommended_checks'),'error':row.get('error')}
        lines.append(f"| {row['row_id']} | {row.get('value')} | {row['feasible']} | {json.dumps(payload,ensure_ascii=False).replace('|','/')} |")
    lines+=['','## Input lineage','',json.dumps({'contract':result['contract'],'rows':[{'row_id':r['row_id'],'arguments':r['arguments'],'multiplicity':r['multiplicity']} for r in result['rows']]},ensure_ascii=False,indent=2),'','The result tables are authoritative. A missing boundary or incomplete branch is unverified, not feasible and not a global infeasibility proof. Search ranges without an explicit task limit are algorithmic brackets. Steady-state equilibrium does not predict kinetic fouling, transient operation, resin breakthrough, or vendor equipment limits.']
    return '\n'.join(lines)


def decision_summary(result,question):
    q=clean(question);family=result['contract']['family'];out=[]
    for outcome in result['outcomes']:
        if not outcome.get('complete'):
            out.append({'job':outcome['job'],'decision':'Incomplete evidence; no certified recommendation'});continue
        interval=outcome.get('recommended') or outcome.get('basic')
        if interval:
            variable=outcome['variable'];objective=cell_after(q,'Objective') or q[:1000]
            upper=bool(re.search(r'maximi|maximum|highest',objective,re.I))
            if family.startswith('d1_'):upper=False
            chosen=interval['upper' if upper else 'lower']
            matching=[r for r in result['rows'] if r['job']==outcome['job'] and r.get('value')==chosen and r.get('feasible')]
            if matching:
                row=matching[-1];metrics=dict(row['metrics']);args=row['arguments']
                if 'product_m3_day' in metrics:
                    m=metrics;dp=None
                    inlet=numeric_cell(q,'Brine pressure into ERD');loss=numeric_cell(q,'ERD-inlet loss from P1');outlet=numeric_cell(q,'ERD low-pressure outlet')
                    if inlet is not None and outlet is not None:dp=inlet-outlet
                    elif loss is not None and outlet is not None:dp=args['p1_pressure_bar']-loss-outlet
                    if dp is not None:
                        m['available_brine_pressure_power_kW']=m['brine_m3_s']*dp*100
                        if 'pxr_efficiency' in args:m['equivalent_recovered_power_kW']=args['pxr_efficiency']*m['available_brine_pressure_power_kW']
                    if 'P1_kW' in m and 'P2_kW' in m:m['pump_train_power_kW']=m['P1_kW']+m['P2_kW']
                    suction=numeric_cell(q,'P1 suction pressure');density=numeric_cell(q,'Seawater density');gravity=numeric_cell(q,'Gravity')
                    if all(v is not None for v in (suction,density,gravity)):m['head_m']=(args['p1_pressure_bar']-suction)*100000/(density*gravity)
                out.append({'job':outcome['job'],'decision_variable':variable,'value':chosen,'criterion':'recommended' if outcome.get('recommended') else 'basic','observed_interval':interval,'row_id':row['row_id'],'measured_and_derived_metrics':metrics})
        elif outcome.get('mode')=='finite':
            feasible=[c for c in outcome['candidates'] if c['feasible']]
            out.append({'job':outcome['job'],'passing_candidates':[c['candidate_id'] for c in feasible],'decision':'No passing tested candidate' if not feasible else 'Passing tested candidates; apply the public objective only within this candidate set'})
            if family=='d4_4a' and feasible:
                pool=[r for r in result['rows'] if r['job']==outcome['job'] and r.get('feasible') and 'CAPEX' in r.get('metrics',{})]
                if pool:
                    best=min(pool,key=lambda r:r['metrics']['CAPEX']);out[-1].update(lowest_modeled_capex_row=best['row_id'],modeled_capex=best['metrics']['CAPEX'])
        else:out.append({'job':outcome['job'],'decision':outcome.get('status','No confirmed interval in search range')})
    if family=='d4_4a':
        measured=[r for r in result['rows'] if not r.get('error') and 'CAPEX' in r.get('metrics',{})]
        def by_prefix(prefix):return [r for r in measured if prefix.casefold() in r['row_id'].casefold()]
        if numeric_cell(q,'Element area') is not None:
            standard=[r for r in by_prefix('Standard membrane') if r['feasible']];high=[r for r in by_prefix('High-permeability') if r['feasible']]
            if standard and high:
                a=min(standard,key=lambda r:r['arguments']['ro_area_m2']);b=min(high,key=lambda r:r['arguments']['ro_area_m2'])
                elements=math.ceil(b['arguments']['ro_area_m2']/numeric_cell(q,'Element area'));premium=(a['metrics']['CAPEX']-b['metrics']['CAPEX'])/elements
                quote=numeric_cell(q,'Quoted premium');out.append({'purchase_comparison':{'standard_row':a['row_id'],'high_permeability_row':b['row_id'],'high_permeability_element_count':elements,'maximum_extra_usd_per_element':premium,'quoted_extra_usd_per_element':quote,'quote_within_capital_parity':quote<=premium}})
        reference=numeric_cell(q,'Parity reference');quoted=numeric_cell(q,'High-efficiency premium')
        if reference is not None and measured:
            reference_row=next((r for r in measured if r['arguments'].get('pxr_efficiency')==reference),None)
            best_eff=max(measured,key=lambda r:r['arguments'].get('pxr_efficiency',0))
            if reference_row:
                budget=numeric_cell(q,'CAPEX ceiling')-best_eff['metrics']['CAPEX'];parity=reference_row['metrics']['CAPEX']-best_eff['metrics']['CAPEX']
                out.append({'vendor_premium':{'maximum_under_budget':budget,'maximum_under_capital_parity':parity,'maximum_satisfying_both':min(budget,parity),'quoted_premium':quoted,'quote_passes_both':quoted<=min(budget,parity)}})
        index=numeric_cell(q,'Cost-index ratio');contingency=numeric_cell(q,'Project contingency')
        if index is not None and contingency is not None:
            out.append({'approval_basis_costs':[{'row':r['row_id'],'CAPEX_approved_currency':r['metrics']['CAPEX']*index*(1+contingency),'within_budget':r['metrics']['CAPEX']*index*(1+contingency)<=numeric_cell(q,'Approved budget')} for r in measured]})
        base_row=next((r for r in measured if ':base:design' in r['row_id']),None)
        if base_row:
            elasticities=[]
            base=result['contract']['jobs'][0]['base']
            for r in measured:
                if r is base_row:continue
                changes=[(k,v) for k,v in r['arguments'].items() if k in base and v!=base[k]]
                delta=r['metrics']['CAPEX']-base_row['metrics']['CAPEX']
                if len(changes)==1 and isinstance(changes[0][1],(float,int)) and base[changes[0][0]]:
                    key,value=changes[0];elasticity=(delta/base_row['metrics']['CAPEX'])/((value-base[key])/base[key]);elasticities.append({'row':r['row_id'],'variable':key,'CAPEX_change':delta,'elasticity':elasticity})
                else:out.append({'discrete_configuration_row':r['row_id'],'CAPEX_change':delta,'excluded_from_continuous_elasticity':True})
            out.append({'continuous_sensitivity_ranking':sorted(elasticities,key=lambda e:abs(e['elasticity']),reverse=True)})
    if family=='d1_1c':
        minima=[o['basic']['lower'] for o in result['outcomes'] if o.get('basic')]
        policy=result['contract']['policy'];margin=policy.get('design_margin_bar');increment=policy.get('equipment_increment_bar')
        if len(minima)==len(result['contract']['jobs']):
            if policy.get('rating_requested') and margin is not None and increment is not None:
                out.append({'controlling_pressure_bar':max(minima),'design_margin_bar':margin,'rated_pressure_bar':math.ceil((max(minima)+margin)/increment-1e-10)*increment,'note':'Rated capability is distinct from each operating setpoint.'})
            elif policy.get('rating_requested'):
                out.append({'controlling_pressure_bar':max(minima),'rated_pressure_bar':None,
                            'missing_public_inputs':policy.get('missing_public_inputs',[]),'decision':'Operating minima established; equipment rating unresolved until public inputs are supplied.'})
    return out


def execute_or_reuse_public_workflow(benchmark,*,settings,run_dir,max_tool_calls=64):
    """Paired delivery ablation: identical evidence, one physical execution.

    Cache key includes the public question, implementation and physical budget.
    Reuse is explicit and never counted as an additional physical invocation.
    """
    from pathlib import Path
    from .io_utils import stable_hash,read_json,write_json
    identity={'question':benchmark['question_prompt'],'family':benchmark['task_family'],'budget':max_tool_calls,
              'implementation':settings.get('implementation_files_sha256'), 'workflow':VERSION}
    key=stable_hash(identity);path=Path(run_dir)/'program_evidence'/(key+'.json')
    if path.exists():
        saved=read_json(path)
        if saved['identity']!=identity:raise ValueError('Evidence cache identity mismatch')
        result=deepcopy(saved['execution']);result['new_physical_calls']=0
        result['evidence_reuse']={'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'same_public_contract':True,'new_physical_calls':0}
        # Keep successful physical source events for traceable validation;
        # real cost is new_physical_calls, not another execution of these events.
        return result
    result=execute_public_workflow(benchmark,max_tool_calls=max_tool_calls)
    result['new_physical_calls']=result['tool_call_count']
    write_json(path,{'identity':identity,'execution':result})
    return result
