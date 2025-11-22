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

#_________________KKT & KKT RESULT DISPLAYING_____________________

#_________________DIAGNOSTICS & OPTIMISATION______________________

#__________________DISPLAYING RESULT______________________________

#_____________________MAIN FUNCTION_______________________________
