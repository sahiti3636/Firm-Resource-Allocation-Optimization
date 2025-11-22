"""
Firm Resource Allocation Optimization (ECOS-forced)
Team: Cute Force
Members: Potini Sahiti, Navya Sharma, D. Sreeteja
"""
#____________________ECOS CHECK___________________________________
def ensure_ecos_available():
    installed = cp.installed_solvers()
    print("CVXPY installed solvers:", installed)
    if "ECOS" not in installed:
        print("\nERROR: ECOS is not available to CVXPY in this environment.")
        print("Install it in the same env where you're running this script:")
        print("  pip install ecos")
        print("or (conda):")
        print("  conda install -c conda-forge ecos")
        print("\nAfter install, re-run the script in the same Python interpreter.")
        sys.exit(1)

#____________QUIZ & MAPPING QUIZ PARAMETERS_______________________
def firm_quiz():
    """Interactive quiz to capture firm characteristics"""
    print("=" * 50)
    print("FIRM RESOURCE ALLOCATION QUIZ")
    print("=" * 50 + "\n")
    # (same questions as before)
    print("1. What is your firm's stage?")
    print("   a) Early-stage startup\n   b) Growth phase\n   c) Mature/established")
    stage = input("Answer (a/b/c): ").strip().lower()
    stage_score = {'a': 1, 'b': 2, 'c': 3}.get(stage, 2)

    print("\n2. What is your growth ambition?")
    print("   a) Conservative/stable\n   b) Moderate growth\n   c) Aggressive expansion")
    growth = input("Answer (a/b/c): ").strip().lower()
    growth_score = {'a': 1, 'b': 2, 'c': 3}.get(growth, 2)

    print("\n3. Fundraising status?")
    print("   a) Bootstrapped\n   b) Some funding secured\n   c) Well-funded")
    funding = input("Answer (a/b/c): ").strip().lower()
    funding_score = {'a': 1, 'b': 2, 'c': 3}.get(funding, 2)

    print("\n4. Labour availability in your region?")
    print("   a) Scarce/expensive\n   b) Moderate\n   c) Abundant/affordable")
    labour = input("Answer (a/b/c): ").strip().lower()
    labour_score = {'a': 1, 'b': 2, 'c': 3}.get(labour, 2)

    print("\n5. Material cost situation?")
    print("   a) High volatility\n   b) Stable\n   c) Declining/favorable")
    materials = input("Answer (a/b/c): ").strip().lower()
    material_score = {'a': 1, 'b': 2, 'c': 3}.get(materials, 2)

    print("\n6. Current operational efficiency?")
    print("   a) Needs improvement\n   b) Average\n   c) Highly efficient")
    efficiency = input("Answer (a/b/c): ").strip().lower()
    efficiency_score = {'a': 1, 'b': 2, 'c': 3}.get(efficiency, 2)

    print("\n7. Marketing importance for your business?")
    print("   a) Critical\n   b) Important\n   c) Secondary")
    marketing = input("Answer (a/b/c): ").strip().lower()
    marketing_score = {'a': 3, 'b': 2, 'c': 1}.get(marketing, 2)

    print("\n8. Total capital/budget available (in thousands)?")
    capital = float(input("Amount: "))

    return {
        'stage': stage_score,
        'growth': growth_score,
        'funding': funding_score,
        'labour': labour_score,
        'materials': material_score,
        'efficiency': efficiency_score,
        'marketing': marketing_score,
        'capital': capital
    }

def map_parameters(quiz_results):
    base_productivity = quiz_results['efficiency'] * 0.4
    p = np.array([
        base_productivity + 0.3,
        base_productivity + 0.25,
        base_productivity + 0.2
    ])

    c = np.array([
        5.0 - quiz_results['labour'] * 0.5,
        4.0 - quiz_results['materials'] * 0.4,
        3.0
    ])

    growth_factor = quiz_results['growth'] * 0.15
    Q = np.diag([growth_factor, growth_factor * 0.8, growth_factor * 0.6])
    alpha = quiz_results['marketing'] * 2.5
    principal = quiz_results['capital'

    return {'p': p, 'c': c, 'Q': Q, 'alpha': alpha, 'principal': principal}



#_________________KKT & KKT RESULT DISPLAYING_____________________

#_________________DIAGNOSTICS & OPTIMISATION______________________

#__________________DISPLAYING RESULT______________________________

#_____________________MAIN FUNCTION_______________________________
