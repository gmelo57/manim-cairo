from manimlib.imports import *


class TFC2(Scene):
    CONFIG = {
        'camera_config': {'background_color': WHITE},

    }
    def construct(self):
        tfc1 = TexMobject(r'\frac{\mathrm{d} }{\mathrm{d} x} \int_{a}^{x}f(t)dt = f(x)')
        tfc1.set_color(BLACK)
        tfc1.shift(UP * 1)
        tfc2 = TexMobject(r' \int_{a}^{b}f(x)dx = F(b) - F(a)')
        tfc2.set_color(BLACK)
        tfc2.shift(DOWN * 1)
        tfc0 = TextMobject('$Teorema$ $Fundamental$ $do$ $C\\acute{a}lculo$')
        #        tfc0 = TextMobject('\\mathit{Teorema Fundamental do Cálculo}')
        self.play(Write(tfc1))
        self.play(Write(tfc2))
        self.play(tfc1.shift, UP * 0.8, tfc2.shift, DOWN * 0.8)
        self.play(Write(tfc0))
        self.wait()


class Boyle(Scene):
    def construct(self):
        bl = TexMobject('P \\propto \\frac{1}{V}')
        bl.scale(1.2)
        self.play(Write(bl))
        self.play(ShowCreation(SurroundingRectangle(bl)))
        self.wait(2)


class Voltaire(Scene):
    def construct(self):
        n = TextMobject(' $``Newton$ $\\acute{e}$ $o$ $maior$ $homem$ $que$ $j\\acute{a}$ $existiu!"$')
        n.scale(0.9)
        #        t = TexMobject(' \\say {Here, a quotation is written and even some quotations are possible} ')
        self.play(Write(n))
        self.wait(3)


class Note3(Scene):
    CONFIG = {
        'camera_config': {'background_color': WHITE},

    }
    def construct(self):
        nt = TextMobject('Notação para derivada primeira:')
        nt.shift(UP * 2.2)
        newt = TextMobject('Newton')
        newt.shift(UP * 0.8, RIGHT * 2.5)
        leib = TextMobject('Leibniz')
        leib.shift(UP * 0.8, LEFT * 2.5)
        func = TexMobject('y = f(x)')
        func.set_color(BLACK)
        func2 = func.copy()
        func.next_to(newt, DOWN * 2)
        func2.next_to(leib, DOWN * 2)
        dx = TexMobject(r'\frac{\mathrm{d}y }{\mathrm{d} x}')
        dx.next_to(func2, DOWN)
        dx.set_color(BLACK)
        y = TexMobject('\\dot{y}')
        y.set_color(BLACK)
        y.next_to(func, DOWN)
        self.play(Write(nt))

        self.play(Write(newt), Write(leib))
        self.play(Write(func), Write(func2))
        self.wait()
        self.play(ReplacementTransform(func2.copy(), dx), ReplacementTransform(func.copy(), y))
        self.wait(2)


class Hooke(Scene):
    def construct(self):
        h = TexMobject(r'\vec{F} = k \vec{x}')
        h.scale(1.2)
        self.play(Write(h))
        self.play(ShowCreation(SurroundingRectangle(h)))
        self.wait(2)


class Sum(Scene):
    def construct(self):
        s1 = TexMobject(r'1 + \frac{1}{3} + \frac{1}{6} + \frac{1}{10} + \frac{1}{15} + \frac{1}{21} ...')
        s1.shift(UP)
        s2 = TexMobject(r'\sum_{n=1}^{\infty} \frac{2}{n(n+1)} = 2')
        s2.shift(DOWN)
        self.play(Write(s1))
        self.wait()
        self.play(Write(s2))
        self.wait(2)


class New(Scene):
    def construct(self):
        n = TextMobject(' $``Newton$ $n\\tilde{a}o$ $se$ $limita$ $aos$ $s\\acute{\\iota}mbolos!"$')
        n.scale(0.9)
        self.play(Write(n))
        self.wait(3)


class Det(VectorScene):
    def construct(self):
        #        m = Matrix([['a','b'],['c','d'], get_det_text()],)
        n = TexMobject('\\begin{pmatrix}a & b \\\c &  d\\end{pmatrix}')
        det = TexMobject('ad - bc')
        det.shift(DOWN*0.5)
        self.play(Write(n))
        self.play(n.shift,UP)
        self.play(ReplacementTransform(n.copy(),det))
        self.wait()

class Ani(Scene):
    def construct(self):
        a = TextMobject('''Anni mirabiles \n
                           (1665-1667)''')
        a.scale(0.9)
        self.play(Write(a))
        self.wait(3)

