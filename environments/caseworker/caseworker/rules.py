"""SNAP eligibility and benefit rules (simplified federal rules, FFY 2026 figures).

Figures are the FFY 2026 cost-of-living adjustments for the 48 contiguous states and DC
(effective Oct 1, 2025 - Sep 30, 2026), as restated in Maryland DHS Action Transmittal 26-05.
Utility allowances (SUA/LUA/telephone) are state-set; we use Maryland's FFY 2026 values.
Structure follows 7 CFR 273.9 (income and deductions) and 273.10 (benefit computation).

This is a teaching/benchmark ruleset, not legal advice: it omits proration, expedited service,
work rules, child support deductions, homeless shelter deduction, and state options like BBCE.

`determine(case, mistakes=...)` computes the correct decision, or the decision a caseworker
would reach under one or more common mistakes (used to label why an agent went wrong).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

GROSS_LIMIT = {1: 1696, 2: 2292, 3: 2888, 4: 3483, 5: 4079, 6: 4675, 7: 5271, 8: 5867}  # 130% FPL
NET_LIMIT = {1: 1305, 2: 1763, 3: 2221, 4: 2680, 5: 3138, 6: 3596, 7: 4055, 8: 4513}  # 100% FPL
MAX_ALLOTMENT = {1: 298, 2: 546, 3: 785, 4: 994, 5: 1183, 6: 1421, 7: 1571, 8: 1789}
EXTRA = {"gross": 596, "net": 459, "allotment": 218}  # each member beyond 8
STANDARD_DEDUCTION = {1: 209, 2: 209, 3: 209, 4: 223, 5: 261}  # 6+: 299
SHELTER_CAP = 744
SUA, LUA, PHONE = 572, 350, 40
MIN_BENEFIT = 24
RESOURCE_LIMIT, RESOURCE_LIMIT_ED = 3000, 4500
MEDICAL_THRESHOLD = 35
EARNED_DEDUCTION = 0.20
FREQ = {"weekly": 4.3, "biweekly": 2.15, "semimonthly": 2.0, "monthly": 1.0}
COUNTABLE_RESOURCES = ("checking", "savings", "cash")
HOUSING = ("rent", "mortgage", "property_tax", "homeowners_insurance")

MISTAKES = (
    "no_earned_deduction",          # forgot the 20% earned income deduction
    "ignored_elderly_disabled",     # applied gross test / shelter cap / $3,000 limit / no medical to an E/D household
    "counted_student_earnings",     # counted wages of a student under 22
    "counted_moved_out_member",     # kept a member (and income) who left before the application date
    "dropped_future_mover",         # removed a member whose move-out is after the application date
    "counted_duplicate_income",     # counted a record marked as a duplicate upload
    "wrong_pay_frequency",          # weekly x4 / biweekly x2 instead of 4.3 / 2.15
    "counted_excluded_resources",   # counted a vehicle or retirement account
    "actual_utilities",             # used the utility bill instead of the standard allowance
    "rounded_30pct_down",           # rounded 30% of net income down instead of up
    "no_minimum_benefit",           # gave a 1-2 person household less than $24
    "ignored_categorical",          # ran income/resource tests on an all-SSI/TANF household
    "denied_missing_verification",  # denied instead of pending when verification is missing
)


def _by_size(table: dict[int, int], n: int, extra: int) -> int:
    return table[n] if n <= 8 else table[8] + (n - 8) * extra


def standard_deduction(n: int) -> int:
    return STANDARD_DEDUCTION.get(n, 299)


@dataclass
class Decision:
    decision: str  # approved | denied | pending_verification
    benefit: int = 0
    reason: str = ""  # denials: resources | gross_income | net_income | no_benefit
    work: dict = field(default_factory=dict)  # intermediate amounts, for debugging and judges


def _r2(x: float) -> float:
    return round(x + 1e-9, 2)


def determine(case: dict, mistakes: frozenset[str] | set[str] = frozenset()) -> Decision:
    """`case` holds the case record plus its members, income, expenses, resources, verifications."""
    m = set(mistakes)
    app = case["application_date"]
    if any(v["status"] != "verified" for v in case["verifications"]):
        if "denied_missing_verification" in m:
            return Decision("denied", reason="missing_verification")
        return Decision("pending_verification")

    def in_household(mem: dict) -> bool:
        out = mem.get("moved_out_date") or ""
        if not out:
            return True
        if out < app:
            return "counted_moved_out_member" in m
        return "dropped_future_mover" not in m

    members = {k: v for k, v in case["members"].items() if in_household(v)}
    n = len(members)
    ed = any(v["age"] >= 60 or v.get("receives") in ("SSI", "SSDI") for v in members.values())
    if "ignored_elderly_disabled" in m:
        ed = False
    categorical = all(v.get("receives") in ("SSI", "TANF") for v in members.values())
    if "ignored_categorical" in m:
        categorical = False

    earned = unearned = 0.0
    for inc in case["income"]:
        if inc["member_id"] not in members:
            continue
        if inc.get("duplicate_of") and "counted_duplicate_income" not in m:
            continue
        mem = members[inc["member_id"]]
        if inc["type"] == "wages" and mem.get("student") and mem["age"] < 22 and "counted_student_earnings" not in m:
            continue
        f = FREQ[inc["frequency"]]
        if "wrong_pay_frequency" in m:
            f = {"weekly": 4.0, "biweekly": 2.0}.get(inc["frequency"], f)
        amt = _r2(inc["amount"] * f)
        if inc["type"] == "wages":
            earned += amt
        else:
            unearned += amt
    earned, unearned = _r2(earned), _r2(unearned)
    gross = _r2(earned + unearned)
    w = {"household_size": n, "elderly_disabled": ed, "categorical": categorical, "gross": gross}

    countable = COUNTABLE_RESOURCES + (("vehicle", "retirement") if "counted_excluded_resources" in m else ())
    resources = _r2(sum(r["balance"] for r in case["resources"] if r["type"] in countable))
    w["resources"] = resources
    if not categorical:
        if resources > (RESOURCE_LIMIT_ED if ed else RESOURCE_LIMIT):
            return Decision("denied", reason="resources", work=w)
        if not ed and gross > _by_size(GROSS_LIMIT, n, EXTRA["gross"]):
            return Decision("denied", reason="gross_income", work=w)

    earned_ded = 0.0 if "no_earned_deduction" in m else _r2(earned * EARNED_DEDUCTION)
    dep = _r2(sum(e["amount"] for e in case["expenses"] if e["type"] == "dependent_care"))
    med = 0.0
    if ed:
        ed_ids = {k for k, v in members.items() if v["age"] >= 60 or v.get("receives") in ("SSI", "SSDI")}
        med_costs = sum(e["amount"] for e in case["expenses"] if e["type"] == "medical" and e.get("member_id") in ed_ids)
        med = _r2(max(med_costs - MEDICAL_THRESHOLD, 0))
    adjusted = _r2(max(gross - earned_ded - standard_deduction(n) - dep - med, 0))

    types = {e["type"] for e in case["expenses"]}
    if "actual_utilities" in m:
        util = sum(e["amount"] for e in case["expenses"] if e["type"] in ("heating_cooling", "utilities_no_heat", "phone"))
    elif "heating_cooling" in types:
        util = SUA
    elif "utilities_no_heat" in types:
        util = LUA
    elif "phone" in types:
        util = PHONE
    else:
        util = 0
    shelter = _r2(sum(e["amount"] for e in case["expenses"] if e["type"] in HOUSING) + util)
    excess = _r2(max(shelter - adjusted / 2, 0))
    if not ed:
        excess = min(excess, SHELTER_CAP)
    net = _r2(max(adjusted - excess, 0))
    w.update(earned_deduction=earned_ded, standard_deduction=standard_deduction(n), dependent_care=dep,
             medical_deduction=med, adjusted_income=adjusted, shelter_costs=shelter, excess_shelter=excess, net=net)

    if not categorical and net > _by_size(NET_LIMIT, n, EXTRA["net"]):
        return Decision("denied", reason="net_income", work=w)
    thirty = math.floor(_r2(net * 0.3)) if "rounded_30pct_down" in m else math.ceil(_r2(net * 0.3))
    benefit = _by_size(MAX_ALLOTMENT, n, EXTRA["allotment"]) - thirty
    if n <= 2 and benefit < MIN_BENEFIT and "no_minimum_benefit" not in m:
        benefit = MIN_BENEFIT
    if benefit <= 0:
        return Decision("denied", reason="no_benefit", work=w)
    return Decision("approved", benefit=int(benefit), work=w)


POLICY_MANUAL = {
    "POL-100": ("Scope", "These rules decide SNAP applications in this office. Apply them exactly, in order. "
                "All dollar figures are FFY 2026 values for the 48 contiguous states and DC."),
    "POL-110": ("Verification", "If ANY verification record for the case has status other than 'verified', do not "
                "decide the case: set decision 'pending_verification' and send the applicant a request that "
                "names every missing item. Do not deny a case for missing verification."),
    "POL-120": ("Household", "The household is every member listed for the case EXCEPT members whose moved_out_date "
                "is before the application_date. A member with a later (planned) moved_out_date is still in the "
                "household. Household size = number of household members."),
    "POL-130": ("Elderly/disabled (E/D)", "A household is E/D if any household member is age 60 or older, or "
                "receives SSI or SSDI. E/D households: no gross income test; resource limit $4,500; medical "
                "deduction allowed; excess shelter deduction is NOT capped."),
    "POL-140": ("Categorical eligibility", "If EVERY household member receives SSI or TANF, skip the resource, "
                "gross income and net income tests. Still compute the benefit."),
    "POL-150": ("Income", "Convert each income record to monthly: weekly x 4.3, biweekly x 2.15, semimonthly x 2, "
                "monthly x 1; round each to cents. Ignore records with a duplicate_of value (duplicate uploads). "
                "Ignore wages of a member who is a student and under age 22. Wages are earned income; all other "
                "types are unearned. Gross income = earned + unearned."),
    "POL-160": ("Resources", "Countable resources = checking + savings + cash. Vehicles and retirement accounts "
                "are excluded. Limit: $3,000, or $4,500 for E/D households. Over the limit: deny, reason 'resources'."),
    "POL-170": ("Gross income test", "Non-E/D households: deny (reason 'gross_income') if gross income exceeds the "
                "130% limit: 1:$1,696 2:$2,292 3:$2,888 4:$3,483 5:$4,079 6:$4,675 7:$5,271 8:$5,867, "
                "+$596 per member beyond 8."),
    "POL-180": ("Deductions", "Earned income deduction = 20% of earned income. Standard deduction by size: "
                "1-3:$209 4:$223 5:$261 6+:$299. Dependent care: actual monthly cost. Medical (E/D households "
                "only): medical costs of E/D members above $35 per month. Adjusted income = gross minus these "
                "deductions (not below 0). Round to cents at each step."),
    "POL-190": ("Shelter", "Shelter costs = rent or mortgage + property tax + homeowners insurance + ONE utility "
                "allowance (never the actual utility bill): $572 SUA if the household pays heating or cooling; "
                "else $350 LUA if it pays other utilities; else $40 if it pays only a telephone bill. Excess "
                "shelter = shelter costs minus half of adjusted income (not below 0), capped at $744 unless the "
                "household is E/D. Net income = adjusted income minus excess shelter (not below 0)."),
    "POL-200": ("Net income test", "Deny (reason 'net_income') if net income exceeds the 100% limit: 1:$1,305 "
                "2:$1,763 3:$2,221 4:$2,680 5:$3,138 6:$3,596 7:$4,055 8:$4,513, +$459 per member beyond 8."),
    "POL-210": ("Benefit", "Benefit = maximum allotment minus 30% of net income, where 30% of net income is "
                "rounded UP to the next whole dollar. Maximum allotments: 1:$298 2:$546 3:$785 4:$994 5:$1,183 "
                "6:$1,421 7:$1,571 8:$1,789, +$218 per member beyond 8. Households of 1-2 people that pass the "
                "tests get at least $24. Otherwise, if the benefit is $0 or less, deny with reason 'no_benefit'."),
    "POL-220": ("Recording and notices", "On each processed case set: decision (approved / denied / "
                "pending_verification), monthly_benefit (whole dollars; 0 unless approved) and denial_reason "
                "(resources / gross_income / net_income / no_benefit; empty unless denied). Send exactly one notice "
                "email per processed case to the applicant's email. The notice must include the case id and the "
                "decision; for approvals the monthly benefit as $N; for denials the denial reason code; for "
                "pending cases every missing verification item. Approval and denial notices must state that the "
                "applicant has the right to request a fair hearing."),
}
