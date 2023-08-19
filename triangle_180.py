from manimlib.imports import *

class tri(Scene):
    CONFIG = {
        'alfa': -1.75 * PI / 6,

    }

    def construct(self):
        alfa = ValueTracker(self.alfa)
        du = UP * 0.5
        lbaixo = Line([-3.015, 0.5, 0], [2.015, 0.5, 0])
        lesq = Line([-3, 0.5, 0], [1.019, 0.5, 0])
        ld = Line([-0.525, 0.5, 0], [2, 0.5, 0])
        linec = Line(LEFT * 8, RIGHT * 8)
        linec.next_to(ld, UP, buff=0)
        linec.shift(UP * 2)
        lineb = Line(LEFT * 8, RIGHT * 8)
        lineb.next_to(ld, DOWN, buff=0.01)
        ld.rotate(alfa.get_value(), about_point=[2, 0.5, 0])
        ld.add_updater(
            lambda m: m.set_angle(
                alfa.get_value(),
            )
        )
        lesq.rotate(PI / 6, about_point=[-3, 0.5, 0])
        angbeta = Arc(
            radius=0.7,
            start_angle=0,
            angle=-1.45 * PI / 6,
            color=WHITE
        )

        angbeta.rotate(PI)
        angbeta.shift(UP * 1, RIGHT * 0.9)

        angbeta2 = Arc(
            radius=0.7,
            start_angle=0,
            angle=-1.4 * PI / 6,
            color=WHITE
        )
        angbeta2.shift(UP * 2.5, RIGHT * 0.3)

        angalfa1 = Arc(
            radius=0.7,
            start_angle=0,
            angle=PI / 6,
            color=WHITE
        )
        angalfa1.shift(UP * 0.5, LEFT * 3)

        angalfa2 = Arc(
            radius=0.7,
            start_angle=0,
            angle=1.02 * PI / 6,
            color=BLUE
        )
        angalfa2.shift(UP * 2.13, LEFT * 0.9)
        angalfa2.rotate(PI * 1.04)

        angama = Arc(
            radius=0.7,
            start_angle=0,
            angle=6 * PI / 12,
            color=WHITE
        )
        angama.rotate(PI * 1.225)
        angama.shift(UP * 1.73, LEFT * 0.01)

        linec2 = Line(LEFT * 8, RIGHT * 8)
        linec2.next_to(linec, UP, buff=0)

        lineb2 = Line(LEFT * 8, RIGHT * 8)
        lineb2.next_to(lineb, DOWN, buff=0)

        alfas = TexMobject(r'\alpha', color=WHITE)
        alfas.next_to(angalfa1, RIGHT)
        alfa2 = TexMobject(r'\alpha', color=BLUE)
        alfa2.next_to(angalfa2, LEFT)

        beta = TexMobject(r'\beta', color=WHITE)
        beta.next_to(angbeta, LEFT)
        beta2 = TexMobject(r'\beta', color=RED)
        beta2.next_to(angbeta2, RIGHT)
        beta2.shift(DOWN * 0.1, LEFT * 0.1)

        gama = TexMobject(r'\gamma', color=WHITE)
        gama.next_to(angama, DOWN)

        formula = TexMobject(r'\alpha', '+', r'\beta', '+', r'\gamma', '=', r'\pi')
        #180^{\circ}
        formula.shift(DOWN * 1, LEFT * 0.2)

        self.play(ShowCreation(lesq),
                  ShowCreation(ld),
                  ShowCreation(lbaixo))

#        self.play(ShowCreation(linec2), ShowCreation(lineb2))

        self.play(ShowCreation(angalfa1),
                  Write(alfas),ShowCreation(angbeta),
                  Write(beta),ShowCreation(angama),
                  Write(gama))

#        self.play(ShowCreation(angalfa2),
#                  Write(alfa2))

#        self.play(ShowCreation(angbeta),
#                  Write(beta))

#        self.play(ShowCreation(angbeta2),
#                  Write(beta2))

#        self.play(ShowCreation(angama),
#                  Write(gama))

#        formula[0].set_color(BLUE)
#        formula[2].set_color(RED)
#        formula[4].set_color(YELLOW)

        self.play(Write(formula[1]),
                  Write(formula[3]),
                  Write(formula[5]),
                  Write(formula[6])
                  )

        self.play(ReplacementTransform(alfas.copy(), formula[0]),
                  ReplacementTransform(beta.copy(), formula[2]),
                  ReplacementTransform(gama.copy(), formula[4]),
                  run_time=1.6
                  )

        self.wait(2)
