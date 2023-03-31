import dataclasses
import ltn
import iltn


@dataclasses.dataclass
class LTNOperators:
    And: ltn.fuzzy_ops.And_Prod
    Or: ltn.fuzzy_ops.Or_ProbSum
    Not: ltn.fuzzy_ops.Not_Std
    Implies: ltn.fuzzy_ops.Implies_Reichenbach
    Exists: ltn.fuzzy_ops.Aggreg_pMean
    Forall: ltn.fuzzy_ops.Aggreg_LogProd

@dataclasses.dataclass
class TrapzOperators:
    between: iltn.relations.trapz.operators.Between
    before: iltn.relations.trapz.operators.Before
    after: iltn.relations.trapz.operators.After
    start: iltn.relations.trapz.operators.Start
    end: iltn.relations.trapz.operators.End
    duration: iltn.relations.trapz.operators.Duration

@dataclasses.dataclass
class TrapzRelations:
    contains: iltn.relations.trapz.relations.Contains
    equals: iltn.relations.trapz.relations.Equals
    before: iltn.relations.trapz.relations.Before
    after: iltn.relations.trapz.relations.After
    starts: iltn.relations.trapz.relations.Starts
    overlaps: iltn.relations.trapz.relations.Overlaps
    during: iltn.relations.trapz.relations.During


def get_default_ltn_operators() -> LTNOperators:
    return LTNOperators(
        And=ltn.fuzzy_ops.And_Prod(), Or=ltn.fuzzy_ops.Or_ProbSum(), Not=ltn.fuzzy_ops.Not_Std(),
        Implies=ltn.fuzzy_ops.Implies_Reichenbach(), Exists=ltn.fuzzy_ops.Aggreg_pMean(p=2.),
        Forall=ltn.fuzzy_ops.Aggreg_LogProd()
    )


def get_default_trapz_operators() -> TrapzOperators:
    return TrapzOperators(between=iltn.relations.trapz.operators.Between(),
        before=iltn.relations.trapz.operators.Before(), after=iltn.relations.trapz.operators.After(), 
        start=iltn.relations.trapz.operators.Start(), end=iltn.relations.trapz.operators.End(), 
        duration=iltn.relations.trapz.operators.Duration())


def get_default_trapz_relations(trapz_ops: TrapzOperators, ltn_ops: LTNOperators, beta: float=1.) -> TrapzRelations:
    contains = iltn.relations.trapz.relations.Contains(beta=beta)
    equals = iltn.relations.trapz.relations.Equals(beta=beta)
    before = iltn.relations.trapz.relations.Before(op_before=trapz_ops.before, contains=contains)
    after = iltn.relations.trapz.relations.After(op_after=trapz_ops.after, contains=contains)
    starts = iltn.relations.trapz.relations.Starts(op_start=trapz_ops.start, 
        op_end=trapz_ops.end, equals=equals, before=before)
    overlaps = iltn.relations.trapz.relations.Overlaps(op_start=trapz_ops.start, op_end=trapz_ops.end,
        equals=equals, before=before)
    during = iltn.relations.trapz.relations.During(op_start=trapz_ops.start, 
        op_end=trapz_ops.end, equals=equals, before=before, after=after, fuzzy_and=ltn_ops.And)
    return TrapzRelations(contains=contains, before=before, after=after,
        equals=equals, starts=starts, overlaps=overlaps, during=during)

