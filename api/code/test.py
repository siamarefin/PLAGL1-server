import rpy2.robjects as robjects
from rpy2.rinterface_lib.embedded import R


def create_isolated_r_session():
    r_session = R()
    r_session.initialize()
    return r_session


r_session = create_isolated_r_session()

with r_session:
    robjects.r(f"""
        print('Hello from R!')       
    """)

    