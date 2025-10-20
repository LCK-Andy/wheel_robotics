# ===============================================================
# Symbolic framework for Differential-Drive Robot
# ===============================================================
from sympy import symbols, Function, Matrix, Eq, sin, cos, simplify
from sympy.physics.mechanics import dynamicsymbols


# ===============================================================
# 1️⃣  FixedStandardWheel class  (from the previous discussion)
# ===============================================================
class FixedStandardWheel:
    def __init__(self, alpha, beta, l, r, R, x, y, theta, phi):
        self.alpha = alpha
        self.beta = beta
        self.l = l
        self.r = r
        self.R = R
        self.x = x
        self.y = y
        self.theta = theta
        self.phi = phi

        # Extract time symbol from phi (phi(t))
        self.t = list(phi.atoms(Function))[0].args[0]
        self.dot_phi = phi.diff(self.t)

        # Robot state and velocity in global frame
        self.zeta_I = Matrix([x, y, theta])
        self.dot_zeta_I = self.zeta_I.diff(self.t)

        # Velocity in robot frame
        self.dot_zeta_R = R * self.dot_zeta_I

        # Build constraints
        self.rolling_constraint = self._build_rolling_constraint()
        self.sliding_constraint = self._build_sliding_constraint()

    def _build_rolling_constraint(self):
        α, β, l, r = self.alpha, self.beta, self.l, self.r
        A = Matrix([[sin(α + β), -cos(α + β), -l * cos(β)]])
        lhs = (A * self.dot_zeta_R)[0]
        return Eq(lhs - r * self.dot_phi, 0)

    def _build_sliding_constraint(self):
        α, β, l = self.alpha, self.beta, self.l
        A = Matrix([[cos(α + β), sin(α + β), l * sin(β)]])
        lhs = (A * self.dot_zeta_R)[0]
        return Eq(lhs, 0)

    def constraints(self, simplify_eqs=False):
        if simplify_eqs:
            return simplify(self.rolling_constraint), simplify(self.sliding_constraint)
        return self.rolling_constraint, self.sliding_constraint


# ===============================================================
# 2️⃣  DifferentialDriveRobot class
# ===============================================================
class DifferentialDriveRobot:
    """
    Symbolic model of a differential-drive robot using two fixed standard wheels.
    """

    def __init__(self):
        # Define symbols and variables
        t = symbols("t")
        self.t = t
        self.l, self.r = symbols("l r", real=True)
        self.alpha_R, self.alpha_L = 0, 0
        self.beta_R, self.beta_L = 0, 0  # wheel planes aligned with x-axis

        # Robot pose variables
        self.x, self.y, self.theta = (Function("x")(t),
                                     Function("y")(t),
                                     Function("theta")(t))

        # Wheel rotation variables
        self.phi_R = Function("phi_R")(t)
        self.phi_L = Function("phi_L")(t)

        # Rotation from inertial → robot frame
        self.R = Matrix([
            [cos(self.theta), sin(self.theta), 0],
            [-sin(self.theta), cos(self.theta), 0],
            [0, 0, 1]
        ])

        # Create right and left wheel objects
        self.right_wheel = FixedStandardWheel(
            alpha=0, beta=0, l=self.l, r=self.r, R=self.R,
            x=self.x, y=self.y, theta=self.theta, phi=self.phi_R
        )
        self.left_wheel = FixedStandardWheel(
            alpha=0, beta=0, l=-self.l, r=self.r, R=self.R,
            x=self.x, y=self.y, theta=self.theta, phi=self.phi_L
        )

        # Build complete matrices
        self.build_robot_level_constraints()

    # ---------------------------------------------------------------
    def build_robot_level_constraints(self):
        # Rebuild rolling matrix J1 and lateral constraint C1
        l = self.l
        self.J1 = Matrix([[1, 0, l],
                          [1, 0, -l]])
        self.C1 = Matrix([[0, 1, 0]])

        # Collect wheel angular velocities
        phiR_dot = self.phi_R.diff(self.t)
        phiL_dot = self.phi_L.diff(self.t)
        self.phi_dot = Matrix([phiR_dot, phiL_dot])

        # Build complete equation
        self.J2 = self.r * Matrix.eye(2)
        self.dot_zeta_I = Matrix([self.x.diff(self.t),
                                  self.y.diff(self.t),
                                  self.theta.diff(self.t)])
        self.dot_zeta_R = self.R * self.dot_zeta_I

        # Left-hand side (LHS)
        top = self.J1 * self.dot_zeta_R
        bottom = self.C1 * self.dot_zeta_R
        self.LHS = Matrix.vstack(top, bottom)

        # Right-hand side (RHS)
        top_rhs = self.J2 * self.phi_dot
        bottom_rhs = Matrix.zeros(1, 1)
        self.RHS = Matrix.vstack(top_rhs, bottom_rhs)

        # Full symbolic form of the constraint equation
        self.constraint_equation = Eq(self.LHS, self.RHS)

    # ---------------------------------------------------------------
    def forward_kinematics(self):
        """Solve for body velocities in the robot frame in terms of wheel speeds."""
        phiR_dot, phiL_dot = self.phi_dot
        v_r = self.r / 2 * (phiR_dot + phiL_dot)
        w_r = self.r / (2 * self.l) * (phiR_dot - phiL_dot)
        # Body-frame velocity vector: [x_dot_R, y_dot_R, theta_dot_R]
        v_body = Matrix([v_r, 0, w_r])
        v_body.simplify()
        return v_body

    def show_equations(self):
        print("\n=== Differential Drive Robot Constraint Equation ===")
        print(self.constraint_equation)
        print("\nRolling matrix J1:")
        print(self.J1)
        print("\nSliding constraint C1:")
        print(self.C1)
        print("\nRotation matrix R(theta):")
        print(self.R)


# ===============================================================
# 3️⃣  Demonstration
# ===============================================================
if __name__ == "__main__":
    robot = DifferentialDriveRobot()
    robot.show_equations()

    v_body = robot.forward_kinematics()
    print("\nForward Kinematics (velocities in robot frame):")
    print(v_body)