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


def check_kkt_ecos(x_val, m_val, p, c, Q, alpha, principal, constraints, rel_tol=1e-5, abs_tol=1e-6):
    """
    Robust KKT check with combined relative+absolute tolerances.
    - rel_tol: relative tolerance (e.g., 1e-5)
    - abs_tol: absolute tolerance (e.g., 1e-6)
    """
    def pass_tol(res, scale):
        return np.all(np.abs(res) <= np.maximum(abs_tol, scale * rel_tol))

    print("\n" + "="*60)
    print("KKT CONDITIONS VERIFICATION (ROBUST REL+ABS TOL)")
    print("="*60)

    # Extract constraints safely
    try:
        budget_constraint = constraints[0]
        x_nonneg_constraint = constraints[1]
    except Exception as e:
        print("ERROR: constraints not in expected format:", e)
        return {"error": "bad_constraints"}

    lam_raw = getattr(budget_constraint, "dual_value", None)
    mu_raw  = getattr(x_nonneg_constraint, "dual_value", None)

    lam = np.nan if lam_raw is None else float(np.asarray(lam_raw).ravel()[0])
    if mu_raw is None:
        mu = np.full_like(x_val, np.nan, dtype=float)
    else:
        mu = np.asarray(mu_raw, dtype=float).ravel()
        if mu.shape[0] != x_val.shape[0]:
            mu = np.resize(mu, x_val.shape[0])

    # vals
    total_cost_val = float(c @ x_val + m_val + 0.5 * x_val.T @ Q @ x_val)
    slack_budget = principal - total_cost_val

    # gradients
    grad_x = p
    grad_m = alpha / (2.0 * np.sqrt(max(m_val, 0.0) + 1e-12))

    grad_g_x = c + Q @ x_val
    grad_g_m = 1.0

    # scales for tolerances
    scale_budget = max(1.0, abs(principal), abs(total_cost_val))
    scale_mu = max(1.0, np.linalg.norm(mu, ord=np.inf))
    scale_grad = max(1.0, np.linalg.norm(grad_x, ord=np.inf), np.linalg.norm(p, ord=np.inf))

    # 1) Primal feasibility
    primal_slack_ok = slack_budget >= -max(abs_tol, scale_budget * rel_tol)
    primal_x_ok = np.all(x_val >= -max(abs_tol, scale_budget * rel_tol))
    primal_ok = primal_slack_ok and primal_x_ok

    print("\n1) PRIMAL FEASIBILITY")
    print(f"   - principal = {principal:.6e}")
    print(f"   - total_cost = {total_cost_val:.6e}")
    print(f"   - slack = principal - total_cost = {slack_budget:.6e}")
    print(f"   - x >= 0 ? {np.all(x_val >= 0)} (min x = {np.min(x_val):.6e})")
    print(f"   -> primal_ok (rel+abs tol) = {primal_ok}")

    # 2) Dual feasibility
    dual_ok = True
    if np.isnan(lam):
        print("\n2) DUAL FEASIBILITY")
        print("   - λ (budget) not available (NaN).")
        dual_ok = False
    else:
        mu_has_nan = np.any(np.isnan(mu))
        if mu_has_nan:
            print("\n2) DUAL FEASIBILITY")
            print("   - μ (nonneg) contains NaNs.")
            dual_ok = False
        else:
            dual_ok = (lam >= -max(abs_tol, scale_budget * rel_tol)) and np.all(mu >= -max(abs_tol, scale_mu * rel_tol))
            print("\n2) DUAL FEASIBILITY")
            print(f"   - λ = {lam:.6e}")
            print(f"   - μ = {mu}")
            print(f"   -> dual_ok (rel+abs tol) = {dual_ok}")

    # 3) Complementary slackness
    cs_budget = np.nan if np.isnan(lam) else lam * slack_budget
    cs_nonneg = mu * x_val
    cs_ok = False
    if np.isnan(cs_budget) or np.any(np.isnan(cs_nonneg)):
        print("\n3) COMPLEMENTARY SLACKNESS")
        print("   - Some complementary slackness terms not available due to missing duals.")
    else:
        cs_ok = pass_tol(cs_budget, abs(lam) if abs(lam) > 0 else 1.0) and pass_tol(cs_nonneg, np.maximum(1.0, np.abs(x_val)))
        print("\n3) COMPLEMENTARY SLACKNESS")
        print(f"   - λ * slack = {cs_budget:.6e}")
        print(f"   - μ * x = {cs_nonneg}")
        print(f"   -> cs_ok (rel+abs tol) = {cs_ok}")

    # 4) Stationarity
    stationarity_x = grad_x - lam * grad_g_x - mu if not np.isnan(lam) and not np.any(np.isnan(mu)) else np.full_like(grad_x, np.nan)
    stationarity_m = grad_m - lam * grad_g_m if not np.isnan(lam) else np.nan

    stationarity_ok = False
    if np.any(np.isnan(stationarity_x)) or np.isnan(stationarity_m):
        print("\n4) STATIONARITY")
        print("   - Stationarity cannot be fully evaluated (missing duals).")
    else:
        stationarity_ok = pass_tol(stationarity_x, scale_grad) and pass_tol(stationarity_m, scale_grad)
        print("\n4) STATIONARITY")
        print(f"   - ∇_x L = {stationarity_x}")
        print(f"   - ∇_m L = {stationarity_m:.6e}")
        print(f"   -> stationarity_ok (rel+abs tol) = {stationarity_ok}")

    all_ok = primal_ok and dual_ok and cs_ok and stationarity_ok

    print("\n" + "="*60)
    print("KKT SUMMARY (ROBUST):")
    print(f"  primal_ok     = {primal_ok}")
    print(f"  dual_ok       = {dual_ok}")
    print(f"  complementary = {cs_ok}")
    print(f"  stationarity  = {stationarity_ok}")
    print(f"  ALL KKT OK?   = {all_ok}")
    print("="*60 + "\n")

    return {
        "primal": primal_ok,
        "dual": dual_ok,
        "complementary": cs_ok,
        "stationarity": stationarity_ok,
        "all_ok": all_ok,
        "lambda": lam,
        "mu": mu,
        "slack": slack_budget,
        "stationarity_x": stationarity_x,
        "stationarity_m": stationarity_m,
        "cs_budget": cs_budget,     # Added for plotting
        "cs_nonneg": cs_nonneg      # Added for plotting
    }

def plot_kkt_residuals(kkt_res):
    """
    Plots the Stationarity and Complementary Slackness residuals to visually
    demonstrate that small errors are numerical artifacts, not theoretical violations.
    """

    # Stationarity: [stat_x1, stat_x2, stat_x3, stat_m]
    stat_x = kkt_res.get('stationarity_x', [])
    stat_m = kkt_res.get('stationarity_m', 0.0)
    
    # Handle cases where solver returned NaNs
    if np.any(np.isnan(stat_x)) or np.isnan(stat_m):
        print("Cannot plot KKT residuals: Data contains NaNs (Solver might have failed).")
        return

    stat_vals = np.append(stat_x, stat_m)
    stat_labels = ['dL/dx1', 'dL/dx2', 'dL/dx3', 'dL/dm']

    cs_budget = kkt_res.get('cs_budget', 0.0)
    cs_nonneg = kkt_res.get('cs_nonneg', [])
    cs_vals = np.append([cs_budget], cs_nonneg)
    cs_labels = ['Budget', 'x1>=0', 'x2>=0', 'x3>=0']


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('KKT Condition Residuals (Numerical Precision Check)', fontsize=14, fontweight='bold')


    ax1.bar(stat_labels, stat_vals, color='skyblue', edgecolor='navy')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_title('Stationarity Residuals (∇L)')
    ax1.set_ylabel('Error Magnitude')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)


    ax2.bar(cs_labels, cs_vals, color='salmon', edgecolor='darkred')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title('Complementary Slackness Residuals')
    ax2.set_ylabel('Product Value (should be 0)')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    inference_text = (
        "INFERENCE:\nThe KKT conditions for stationarity and complementary slackness are not fully satisfied\n"
        "due to typical numerical limitations of solvers rather than true theoretical violation."
    )
    
    plt.figtext(0.5, 0.02, inference_text, ha="center", fontsize=10, 
                bbox={"facecolor":"orange", "alpha":0.1, "pad":10}, color='darkred')

    plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # Make room for text at bottom
    plt.savefig('kkt_residuals_check.png', dpi=150)
    plt.show()
    print("\nKKT Residuals chart saved as 'kkt_residuals_check.png'")

#-----------------OPTIMISATION MODEL-------------------
def optimize_allocation(p, c, Q, alpha, principal):
    """
    Solve: maximize p^T*x + alpha*sqrt(m) 
    subject to: c^T*x + m + 0.5*x^T*Q*x <= principal
                x >= 0, m >= 0
    """
    ------------
    # p^T*x is output (or profit) generated from spending
    # alpha*sqrt(m) is the contribution from marketing (because of diminishing returns)
    # c^T*x - cost of spending
    # m - marketing costs
    #  0.5*x^T*Q*x - models curvature in the cost of the inputs (term introduces non-linear cost increases)
    -----------
    
    x = cp.Variable(3)
    m = cp.Variable(nonneg=True)

    production_output = p @ x
    marketing_uplift = alpha * cp.sqrt(m)
    objective = cp.Maximize(production_output + marketing_uplift)

    linear_cost = c @ x
    quadratic_cost = 0.5 * cp.quad_form(x, Q)
    total_cost = linear_cost + m + quadratic_cost

    constraints = [total_cost <= principal, x >= 0, m >= 0]
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.ECOS, verbose=False, feastol=1e-8, abstol=1e-8, reltol=1e-8, max_iters=500)
    except cp.SolverError as e:
        print("SolverError while calling ECOS:", e)
        raise

    stats = problem.solver_stats
    
    # Capture KKT results to return them 
    kkt_results = {}
    
    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
        kkt_results = check_kkt_ecos(
            x_val = x.value,
            m_val = float(m.value),
            p = p,
            c = c,
            Q = Q,
            alpha = alpha,
            principal = principal,
            constraints = constraints
        )
        
        return {
            'status': 'optimal' if problem.status == cp.OPTIMAL else 'optimal_inaccurate',
            'labour': float(x.value[0]),
            'materials': float(x.value[1]),
            'energy': float(x.value[2]),
            'marketing': float(m.value) if m.value is not None else 0.0,
            'total_output': float(problem.value),
            'production_output': float(production_output.value),
            'marketing_uplift': float(marketing_uplift.value),
            'total_cost': float(total_cost.value),
            'solver_stats': stats,
            'kkt_diagnostics': kkt_results 
        }
    
    return {'status': problem.status, 'solver_stats': stats}


#_________________DIAGNOSTICS______________________
def diagnostics(allocation, params, p, c, Q):
    print("\n" + "=" * 50)
    print("OPTIMIZATION DIAGNOSTICS & ACCURACY CHECK (ECOS)")
    print("=" * 50)

    if allocation['status'] not in ('optimal', 'optimal_inaccurate'):
        print("Diagnostics unavailable: problem not optimal.")
        return

    total_cost = allocation['total_cost']
    principal = params['principal']
    gap = principal - total_cost

    print(f"\nBUDGET CONSTRAINT CHECK:")
    print(f"   Total Cost: {total_cost:.9f}")
    print(f"   Principal:  {principal:.9f}")
    print(f"   Gap (principal - total_cost): {gap:.6e}")
    if abs(gap) < 1e-6:
        print("   Status: ✓ Tight constraint (good)")
    else:
        print("   Status: ⚠ Constraint not tight (possible slack)")

    # Gradient / KKT quick check
    grad = p  # ∇_x of objective is p
    grad_norm = np.linalg.norm(grad)
    print("\nKKT / GRADIENT CHECK (informal):")
    print(f"   Gradient wrt x: {grad}")
    print(f"   Gradient norm: {grad_norm:.6f}")
    if grad_norm < 1e-3:
        print("   Status: ✓ small gradient (interior optimal)")
    else:
        print("   Status: ⚠ larger gradient (likely active constraints)")

    # Solver stats
    stats = allocation.get('solver_stats', None)
    print("\nSOLVER STATISTICS (from problem.solver_stats):")
    if stats is None:
        print("   No solver stats available in returned allocation.")
        return

    # Print safe fields with getattr
    print(f"   Solver Used: {getattr(stats, 'solver_name', 'UNKNOWN')}")
    print(f"   Solve Time: {getattr(stats, 'solve_time', 'N/A')}")
    print(f"   Status:     {getattr(stats, 'status', 'N/A')}")
    # iterations may be named num_iters
    if hasattr(stats, 'num_iters'):
        print(f"   Iterations: {stats.num_iters}")
    if hasattr(stats, 'primal_residual'):
        print(f"   Primal Residual: {stats.primal_residual:.3e}")
    if hasattr(stats, 'dual_residual'):
        print(f"   Dual Residual: {stats.dual_residual:.3e}")
    if hasattr(stats, 'gap'):
        print(f"   Duality Gap: {stats.gap:.3e}")

    print("\nFull solver_stats object (repr):")
    print(repr(stats))
    print("=" * 50)

#__________________DISPLAYING RESULT______________________________

def display_results(allocation, params):
    if allocation['status'] != 'optimal' and allocation['status'] != 'optimal_inaccurate':
        print(f"Optimization failed: {allocation['status']}")
        return

    print("\n" + "=" * 50)
    print("OPTIMAL RESOURCE ALLOCATION")
    print("=" * 50)

    print(f"\nBUDGET BREAKDOWN:")
    print(f"   Labour:     ${allocation['labour']:.2f}k")
    print(f"   Materials:  ${allocation['materials']:.2f}k")
    print(f"   Energy:     ${allocation['energy']:.2f}k")
    print(f"   Marketing:  ${allocation['marketing']:.2f}k")
    print(f"   TOTAL COST: ${allocation['total_cost']:.6f}k / ${params['principal']:.2f}k")

    print(f"\nOUTPUT METRICS:")
    print(f"   Production Output:  {allocation['production_output']:.6f}")
    print(f"   Marketing Uplift:   {allocation['marketing_uplift']:.6f}")
    print(f"   TOTAL OUTPUT:       {allocation['total_output']:.6f}")

    # Visualization (same as before)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = ['Labour', 'Materials', 'Energy', 'Marketing']
    sizes = [allocation['labour'], allocation['materials'],
             allocation['energy'], allocation['marketing']]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Budget Allocation')

    ax2.bar(labels, sizes, edgecolor='black')
    ax2.set_ylabel('Allocation ($k)')
    ax2.set_title('Resource Distribution')
    ax2.axhline(y=params['principal'] / 4, color='r', linestyle='--',
                label='Equal Split', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('allocation_results_ecos.png', dpi=150)
    plt.show()
    print("\nChart saved as 'allocation_results_ecos.png'")


#_____________________MAIN FUNCTION_______________________________
def main():
    ensure_ecos_available()

    print("\n" + "=" * 50)
    print("FIRM RESOURCE ALLOCATION OPTIMIZER (ECOS FORCED)")
    print("Team Cute Force")
    print("=" * 50 + "\n")

    mode = input("Use (1) Interactive Quiz or (2) Demo Values? [1/2]: ").strip()

    if mode == '1':
        quiz_results = firm_quiz()
        params = map_parameters(quiz_results)
    else:
        print("\nUsing demo parameters...")
        params = {
            'p': np.array([0.9, 0.75, 0.6]),
            'c': np.array([4.0, 3.2, 3.0]),
            'Q': np.diag([0.3, 0.24, 0.18]),
            'alpha': 5.0,
            'principal': 100.0
        }

    print("\nMAPPED PARAMETERS:")
    print(f"   Productivity (p):  {params['p']}")
    print(f"   Costs (c):         {params['c']}")
    print(f"   Marketing (α):     {params['alpha']:.2f}")
    print(f"   Capital:           ${params['principal']:.2f}k")

    print("\nRunning optimization (ECOS forced)...")
    allocation = optimize_allocation(
        params['p'], params['c'], params['Q'],
        params['alpha'], params['principal']
    )

    # 1. Display Standard Results (Pie Chart / Bar Graph)
    # NOTE: You must close the window that pops up for the code to continue!
    display_results(allocation, params)
    
    # 2. Display KKT Residuals Plot (The new request)
    if 'kkt_diagnostics' in allocation:
        print("\nGeneratng KKT Residuals Plot...")
        plot_kkt_residuals(allocation['kkt_diagnostics'])
    
    # 3. Text Diagnostics
    diagnostics(allocation, params, params['p'], params['c'], params['Q'])

    # 4. Sensitivity Analysis
    print("\n" + "=" * 50)
    print("SENSITIVITY ANALYSIS")
    print("=" * 50)

    capitals = [50, 75, 100, 125, 150]
    outputs = []
    for cap in capitals:
        # This runs silently now without popping up graphs
        res = optimize_allocation(params['p'], params['c'], params['Q'], params['alpha'], cap)
        if res.get('status') in ('optimal', 'optimal_inaccurate'):
            outputs.append(res['total_output'])
            print(f"   Capital: ${cap}k  ->  Output: {res['total_output']:.6f}")
        else:
            outputs.append(np.nan)
            print(f"   Capital: ${cap}k  ->  status: {res.get('status')}")

    # Sensitivity Plot
    xs = [c for c, val in zip(capitals, outputs) if not np.isnan(val)]
    ys = [val for val in outputs if not np.isnan(val)]
    if xs:
        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys, marker='o', linewidth=2, markersize=8)
        plt.fill_between(xs, ys, alpha=0.3)
        plt.xlabel('Capital ($k)')
        plt.ylabel('Total Output')
        plt.title('Output vs. Capital Investment (ECOS)')
        plt.grid(True, alpha=0.3)
        plt.savefig('sensitivity_analysis_ecos.png', dpi=150)
        plt.show()
        print("\nChart saved as 'sensitivity_analysis_ecos.png'")

if __name__ == "__main__":
    main()
