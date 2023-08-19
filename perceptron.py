import numpy as np

from manimlib.imports import *


class Plano(ThreeDScene):
    def construct(self):
        axis_config = {'x_min': -5,
                       'x_max': 5,
                       'y_min': -5,
                       'y_max': 5,
                       'z_min': -3.5,
                       'z_max': 3.75
                       }
        p1 = Dot([1,1,0], color=GREEN).set_shade_in_3d()
        p2 = Dot([-1,1,0], color=ORANGE).set_shade_in_3d()
        p3 = Dot([1, -1, 0], color=ORANGE).set_shade_in_3d()
        p4 = Dot([-1, -1, 0], color=ORANGE).set_shade_in_3d()
        axes = ThreeDAxes(**axis_config)
        self.set_camera_orientation(phi=75*DEGREES, )
        self.add(axes)
        self.begin_ambient_camera_rotation(rate=0.2)
        plane = ParametricSurface(lambda u,v : np.array([0.5*u, -0.5*u + 0.5, v]),
                                  resolution=(15, 32), v_min=-3,v_max=3,u_min=-3,u_max=3,
                                  ).fade(0.1)
        self.play(FadeIn(p1), FadeIn(p2), FadeIn(p3), FadeIn(p4))
        self.wait()
#        self.play(FadeIn(plane))
        t = TextMobject('IA ','Matemática?')
        t[0].set_color(YELLOW), t.shift(UP*2.5, RIGHT*3.5), t.scale(1.5)
#        self.add_fixed_in_frame_mobjects(t)
#        self.add(plane, p1,p2,p3,p4)

        self.wait(7)

class Bin(Scene):
    def construct(self):
        nat = TextMobject('Verdadeiro')
        int = TextMobject('Falso')
        rac = TextMobject('Sim')
        ira = TextMobject('Não')
        real = TextMobject('Positivo')
        complex = TextMobject('Negativo')
        nat.scale(1), nat.set_color(GREEN), nat.shift(UP * 2, LEFT * 2)
        int.scale(1), int.shift(UP * 2, RIGHT * 2), int.set_color(ORANGE)
        rac.scale(1), rac.shift(LEFT * 2), rac.set_color(GREEN)
        ira.scale(1), ira.shift(RIGHT * 2), ira.set_color(ORANGE)
        real.scale(1), real.shift(DOWN * 2, LEFT * 2), real.set_color(GREEN)
        complex.scale(1), complex.shift(DOWN * 2, RIGHT * 2), complex.set_color(ORANGE)
        self.play(Write(nat))
        self.play(ReplacementTransform(nat.copy(), int)), self.play(ReplacementTransform(int.copy(), rac))
        self.play(ReplacementTransform(rac.copy(), ira)), self.play(ReplacementTransform(ira.copy(), real))
        self.play(ReplacementTransform(real.copy(), complex))
        self.wait()


class Net(Scene):
    def construct(self):
        r1 = NeuralNetworkMobject([6, 5, 6, 4, 3, 1])
        r1.scale(1.2)
        r2 = NeuralNetworkMobject([6, 5, 4, 6, 4, 2])
        r2.scale(1.2)
        r3 = NeuralNetworkMobject([7, 5, 7, 4, 3, 2])
        r3.scale(1.2)
        r4 = NeuralNetworkMobject([7, 5, 7, 5, 3, 1])
        r4.scale(1.2)
        self.play(ShowCreation(r1, run_time=5))
#        self.wait(0.5)
        self.play(ReplacementTransform(r1, r2))
#        self.wait(0.8)
        self.play(ReplacementTransform(r2, r3))
        self.play(ReplacementTransform(r3, r4))
        self.wait(2)


class Frank(Scene):
    def construct(self):
        fr = TextMobject('Frank Rosenblatt')
        ano = TextMobject('1928 - 1971')
        ano.next_to(fr, DOWN)
        self.play(FadeIn(fr))
        self.play(FadeIn(ano))
        self.wait(2)


class Ia(Scene):
    def construct(self):
        ia = TextMobject('Inteligência Artificial')
        ia.shift(UP * 2)
        bria = Brace(ia)
        rn = TextMobject('Redes Neurais Artificiais')
        rn.shift(UP * 0.8)
        brn = Brace(rn)
        per = TextMobject('Perceptron')
        per.shift(DOWN * 0.4)
        cla = TextMobject('Classificador linear e binário')
        cla.shift(DOWN * 1.5)
        bper = Brace(per)
        self.play(FadeIn(ia))
        self.wait()
        self.play(FadeIn(bria))
        self.play(FadeIn(rn))
        self.wait()
        self.play(FadeIn(brn))
        self.play(FadeIn(per))
        self.wait()
        self.play(FadeIn(bper))
        self.play(FadeIn(cla))
        self.wait()


class Space(VectorScene):
    def construct(self):
        plane = NumberPlane()
        self.play(ShowCreation(plane, lag_ratio=0.1, run_time=2))

        reta = Line([-5, -5, 0], [5, 5, 0])
        sup = Polygon([-5, -5, 0], [5, 5, 0], [-8, 5, 0], [-8, -5, 0], fill_opacity=0.6, color=YELLOW)
        inf = Polygon([-5, -5, 0], [5, 5, 0], [8, 5, 0], [8, -5, 0], fill_opacity=0.6)
        x1simb = TexMobject('x_{1}')
        x1simb.shift([6.5, -0.5, 0]), x1simb.scale(0.75)

        x2simb = TexMobject('x_{2}')
        x2simb.shift([-0.5, 3.5, 0]), x2simb.scale(0.75)
        x = TexMobject(r'\vec{x} = \langle 1 , x_{1} , x_{2} \rangle ')
        x.shift(LEFT * 3.5, UP * 3)
        w = TexMobject(r'\vec{w} = \langle w_{0} , w_{1} , w_{2} \rangle')
        w.next_to(x, DOWN)
        dotp = TexMobject(r'\vec{x} \vec{w} = ', r' x_{1} w_{1} + x_{2} w_{2} + w_{0} = 0')
        dotp.next_to(w, DOWN)
        posi = TexMobject(r' x_{1} w_{1} + x_{2} w_{2} + w_{0} > 0')
        posi.set_color(YELLOW), posi.shift(DOWN * 0.75, LEFT * 4)
        neg = TexMobject(r' x_{1} w_{1} + x_{2} w_{2} + w_{0} < 0')
        neg.set_color(BLUE), neg.shift(DOWN * 0.75, RIGHT * 3.5)
        #        posi_red = TexMobject('x_{2} > - x_{2} w_{2} - w_{0}')
        self.add_axes(animate=True)
        self.play(Write(x1simb), Write(x2simb))
        self.play(Write(x))
        self.wait()
        self.play(Write(w))
        self.wait()
        self.play(Write(dotp[0]))
        self.wait()
        self.play(Write(dotp[1]))
        self.wait()
        self.play(ShowCreation(reta))
        self.wait()
        self.play(FadeIn(sup), FadeIn(inf), FadeOut(reta))
        self.play(FadeIn(reta))
        self.wait()
        self.play(FadeOut(sup), FadeOut(inf))
        self.wait()
        self.play(ReplacementTransform(dotp[1].copy(), posi), ReplacementTransform(dotp[1].copy(), neg))
        self.wait(2)


import random


class Reta(VectorScene):
    def construct(self):
        x1simb = TexMobject('x_{1}')
        x1simb.shift([6.5, -0.5, 0]), x1simb.scale(0.75)

        x2simb = TexMobject('x_{2}')
        x2simb.shift([-0.5, 3.5, 0]), x2simb.scale(0.75)
        x = TexMobject(r'\vec{x} = \langle  x_{1} , x_{2} \rangle ')
        x.shift(RIGHT * 3.5, UP * 3)
        w = TexMobject(r'\vec{w} = \langle   w_{1} , w_{2} \rangle')
        w.next_to(x, DOWN)
        dotp = TexMobject(r'\vec{x} \vec{w} = ', r' x_{1} w_{1} + x_{2} w_{2} = 0')
        dotp.next_to(w, DOWN)

        x2 = TexMobject(r'\vec{x} = \langle 1 , x_{1} , x_{2} \rangle ')
        x2.shift(RIGHT * 3.5, UP * 3)
        w2 = TexMobject(r'\vec{w} = \langle w_{0} , w_{1} , w_{2} \rangle')
        w2.next_to(x, DOWN)
        dotp2 = TexMobject(r'\vec{x} \vec{w} = ', r' x_{1} w_{1} + x_{2} w_{2} + w_{0} = 0')
        dotp2.next_to(w, DOWN)

        plane = NumberPlane()
        x1 = TexMobject('1')
        x1.shift([1, -0.5, 0]), x1.scale(0.65)
        y1 = TexMobject('1')
        y1.shift([-0.4, 1, 0]), y1.scale(0.65)
        xneg1 = TexMobject('-1')
        xneg1.shift([-1.1, -0.5, 0]), xneg1.scale(0.65)
        yneg1 = TexMobject('-1')
        yneg1.shift([-0.5, -1, 0]), yneg1.scale(0.65)
        self.play(FadeIn(plane))
        self.add_axes(animate=True, x='x_{1}', y='x_{2}')
        self.play(Write(x1), Write(y1), Write(xneg1), Write(yneg1), Write(x1simb), Write(x2simb))
        self.wait()
        self.play(FadeIn(x), FadeIn(w), FadeIn(dotp))
        self.play(FadeIn(Dot([1, 1, 0], color=GREEN)))
        self.play(FadeIn(Dot([-1, 1, 0], color=ORANGE)))
        self.play(FadeIn(Dot([1, -1, 0], color=ORANGE)))
        self.play(FadeIn(Dot([-1, -1, 0], color=ORANGE)))
        graf = Line([-5, 5, 0], [5, -5, 0])
        self.play(ShowCreation(graf))
        self.wait()
        self.play(Rotating(graf, radians=PI, rate_func=rush_from))
        self.wait()
        self.play(ReplacementTransform(x, x2), ReplacementTransform(w, w2))
        self.play(ReplacementTransform(dotp, dotp2))
        self.wait()
        self.play(graf.shift, UP * 0.5)
        self.wait()
        self.play(FadeOut(x2), FadeOut(w2), FadeOut(dotp2))
        for i in range(15):
            a = random.uniform(-5, 5)
            b = random.uniform(-3, 3)
            self.play(ShowCreation(Dot([a, b, 0], color=YELLOW)))

        self.wait(2)


class Relation(Scene):
    def construct(self):
        x1 = TexMobject(r'\langle +1,+1 \rangle')
        x1.shift(UP * 1.5)
        x2 = TexMobject(r'\langle +1,-1 \rangle')
        x2.next_to(x1, DOWN, buff=0.5)
        x3 = TexMobject(r'\langle -1,+1 \rangle')
        x3.next_to(x2, DOWN, buff=0.5)
        x4 = TexMobject(r'\langle -1,-1 \rangle')
        x4.next_to(x3, DOWN, buff=0.5)

        xis1 = TexMobject('\\vec{x}_{1}', '=')
        xis1.shift(UP * 2.5, LEFT * 5.5)
        xis2 = TexMobject('\\vec{x}_{2}', '=')
        xis2.next_to(xis1, DOWN, buff=1.2)
        xis3 = TexMobject('\\vec{x}_{3}', '=')
        xis3.next_to(xis2, DOWN, buff=1.2)
        xis4 = TexMobject('\\vec{x}_{4}', '=')
        xis4.next_to(xis3, DOWN, buff=1.2)

        dot1 = TexMobject('y = sgn(', '\\vec{x}_{1} . \\vec{w}_{1}', '=', r'\,?', ')')
        dot1.shift(UP * 1.5)
        dot2 = TexMobject('y = sgn(', '\\vec{x}_{2} . \\vec{w}_{2}', '=', r'\,?', ')')
        dot2.next_to(dot1, DOWN, buff=0.5)
        dot3 = TexMobject('y = sgn(', '\\vec{x}_{3} . \\vec{w}_{3}', '=', r'\,?', ')')
        dot3.next_to(dot2, DOWN, buff=0.5)
        dot4 = TexMobject('y = sgn(', '\\vec{x}_{4} . \\vec{w}_{4}', '=', r'\,?', ')')
        dot4.next_to(dot3, DOWN, buff=0.5)

        d1 = TexMobject('>', '0')
        d1[0].move_to(dot1[2]), d1[1].move_to(dot1[3])
        d2 = TexMobject('<', '0')
        d2[0].move_to(dot2[2]), d2[1].move_to(dot2[3])
        d3 = TexMobject('<', '0')
        d3[0].move_to(dot3[2]), d3[1].move_to(dot3[3])
        d4 = TexMobject('<', '0')
        d4[0].move_to(dot4[2]), d4[1].move_to(dot4[3])

        self.play(Write(x1))
        self.play(Write(x2))
        self.play(Write(x3))
        self.play(Write(x4))
        self.wait()

        self.play(Write(xis1),
                  Write(xis2),
                  Write(xis3),
                  Write(xis4), )

        self.play(x1.next_to, xis1, RIGHT,
                  x2.next_to, xis2, RIGHT,
                  x3.next_to, xis3, RIGHT,
                  x4.next_to, xis4, RIGHT,
                  rate_func=slow_into,
                  run_time=1.5)

        self.wait()

        self.play(ReplacementTransform(xis1.copy(), dot1[1]),
                  ReplacementTransform(xis2.copy(), dot2[1]),
                  ReplacementTransform(xis3.copy(), dot3[1]),
                  ReplacementTransform(xis4.copy(), dot4[1]))
        self.play(Write(dot1[2:4]),
                  Write(dot2[2:4]),
                  Write(dot3[2:4]),
                  Write(dot4[2:4]))
        self.wait()
        self.play(ReplacementTransform(dot1[2:4], d1),
                  ReplacementTransform(dot2[2:4], d2),
                  ReplacementTransform(dot3[2:4], d3),
                  ReplacementTransform(dot4[2:4], d4))
        self.wait(2)
        self.play(Write(dot1[0:5:2]),
                  Write(dot2[0:5:2]),
                  Write(dot3[0:5:2]),
                  Write(dot4[0:5:2]), )
        self.wait(2)


class Logic(Scene):
    def construct(self):
        tab = TextMobject('Tabela do Operador Lógico "e"')
        tab.shift(UP * 3, RIGHT * 0.5)
        p = TexMobject('p')
        p.shift(LEFT * 1.2 + UP * 1.8), p.scale(1.5)
        q = TexMobject('q')
        q.shift(UP * 1.8), q.scale(1.5)
        pq = TexMobject('p\\wedge q')
        pq.shift(RIGHT * 1.7 + UP * 1.8), pq.scale(1.5)
        l1 = TexMobject('V', 'V', 'V')
        l1[0].next_to(p, DOWN), l1[1].next_to(q, DOWN), l1[2].next_to(pq, DOWN)
        l1.shift(DOWN * 0.5), l1.set_color(GREEN)
        l2 = TexMobject('V', 'F', 'F')
        l2[0].next_to(l1[0], DOWN), l2[1].next_to(l1[1], DOWN), l2[2].next_to(l1[2], DOWN)
        l2.shift(DOWN * 0.5), l2[1:3].set_color(ORANGE), l2[0].set_color(GREEN)
        l3 = TexMobject('F', 'V', 'F')
        l3[0].next_to(l2[0], DOWN), l3[1].next_to(l2[1], DOWN), l3[2].next_to(l2[2], DOWN)
        l3.shift(DOWN * 0.5), l3[0:3:2].set_color(ORANGE), l3[1].set_color(GREEN)
        l4 = TexMobject('F', 'F', 'F')
        l4[0].next_to(l3[0], DOWN), l4[1].next_to(l3[1], DOWN), l4[2].next_to(l3[2], DOWN)
        l4.shift(DOWN * 0.5), l4.set_color(ORANGE)
        #        self.add(p,q,pq,l1,l2,l3,l4)

        l1n = TexMobject('+1', '+1', '+1')
        l1n[0].next_to(p, DOWN), l1n[1].next_to(q, DOWN), l1n[2].next_to(pq, DOWN)
        l1n.shift(DOWN * 0.5), l1n.set_color(GREEN)

        l2n = TexMobject('+1', '-1', '-1')
        l2n[0].next_to(l1n[0], DOWN), l2n[1].next_to(l1n[1], DOWN), l2n[2].next_to(l1n[2], DOWN)
        l2n.shift(DOWN * 0.5), l2n[1:3].set_color(ORANGE), l2n[0].set_color(GREEN)

        l3n = TexMobject('-1', '+1', '-1')
        l3n[0].next_to(l2n[0], DOWN), l3n[1].next_to(l2n[1], DOWN), l3n[2].next_to(l2n[2], DOWN)
        l3n.shift(DOWN * 0.5), l3n[0:3:2].set_color(ORANGE), l3n[1].set_color(GREEN)

        l4n = TexMobject('-1', '-1', '-1')
        l4n[0].next_to(l3n[0], DOWN), l4n[1].next_to(l3n[1], DOWN), l4n[2].next_to(l3n[2], DOWN)
        l4n.shift(DOWN * 0.5), l4n.set_color(ORANGE)
        #        self.add(p, q, pq, l1n, l2n, l3n, l4n)
        self.play(Write(tab))
        self.wait()
        self.play(FadeIn(p)), self.play(FadeIn(q)), self.play(FadeIn(pq))
        self.wait()
        self.play(Write(l1[0:2])), self.wait(), self.play(Write(l1[2]))
        self.wait()
        self.play(Write(l2[0:2])), self.play(Write(l2[2]))
        self.play(Write(l3[0:2])), self.play(Write(l3[2]))
        self.wait()
        self.play(Write(l4[0:2])), self.wait(), self.play(Write(l4[2]))
        self.wait(2)
        self.play(ReplacementTransform(l1, l1n))
        self.play(ReplacementTransform(l2, l2n))
        self.play(ReplacementTransform(l3, l3n))
        self.play(ReplacementTransform(l4, l4n))
        self.wait(2)


class Neuron(Scene):
    def construct(self):
        neuron = ImageMobject("neuron2")
        neuron.scale(2.8)

        sq = Square()
        bae = Brace(sq, LEFT)
        #        bae.next_to(LEFT), bae.rotate(-PI / 2.5), bae.shift(LEFT * 2.2 + UP * 2)
        bae.next_to(neuron, LEFT), bae.shift(RIGHT)
        sq.scale(0.45)
        # self.play(FadeIn(neuron))
        ba = Brace(neuron)
        ba.scale(0.4), ba.shift(UP * 2.5 + RIGHT * 1.3)
        bav = Brace(ba)
        ax = TextMobject('Axônio')
        ax.next_to(bav, DOWN)
        bad = Brace(sq, RIGHT)
        bad.next_to(neuron, RIGHT), bad.shift(LEFT * 1.2)
        ter = TextMobject('''Terminal
                            \n do
                         \n  Axônio''')
        ter.next_to(bad)
        den = TextMobject('Dendritos')
        #        den.next_to(bae, UP), den.shift(UP * 0.1, LEFT * 0.1)
        den.next_to(bae, LEFT)
        l = Line()
        corp = TextMobject('Corpo')

        l.rotate(PI / 3), l.shift(LEFT * 2.6 + DOWN * 1.6)
        corp.next_to(l, DOWN), corp.shift(LEFT * 0.6)
        self.play(FadeIn(neuron))
        self.wait()
        #        self.add(bav, ax, bad, ter, bae, den, l, corp)
        self.play(FadeIn(bae))
        self.play(Write(den)), self.play(Indicate(den))
        self.wait()
        self.play(ShowCreation(l)),
        self.play(Write(corp)), self.play(Indicate(corp))
        self.wait()
        self.play(FadeIn(bav))
        self.play(Write(ax)), self.play(Indicate(ax))
        self.wait()
        self.play(FadeIn(bad))
        self.play(Write(ter)), self.play(Indicate(ter))
        self.wait(2)


class Model(Scene):
    def construct(self):
        net = NeuralNetworkMobject([4, 1, 1], )
        net.scale(3)
        net.label_inputs('x')
        #        net.label_outputs('y')
        #        net.label_hidden_layers('w')
        lb = Line([3.65, -0.25, 0], [3.95, -0.25, 0])
        lm = Line([3.65, -0.25, 0], [4.05, -0.25, 0])
        lm.rotate(PI / 2), lm.next_to(lb, RIGHT + UP, buff=0), lm.shift(LEFT * 0.021)
        lc = Line([3.95, -0.25, 0], [4.25, -0.25, 0])
        lc.next_to(lm, UP + RIGHT, buff=0), lc.shift(LEFT * 0.021)
        sign = VGroup(lb, lc, lm)
        sign.shift(UP * 0.07, LEFT * 0.04)
        #        net.label_outputs_text('y')
        soma = TexMobject('\\sum')
        soma.scale(0.8)
        w0 = TexMobject('w_{0}')
        w0.shift(LEFT * 5, UP * 2.6)
        w1 = TexMobject('w_{1}')
        w1.next_to(w0, DOWN * 6)
        w2 = TexMobject('w_{2}')
        w2.next_to(w1, DOWN * 6)
        wn = TexMobject('w_{n}')
        wn.next_to(w2, DOWN * 6)
        outp = TexMobject(r'''\begin{cases}
                                +1&   \\ 
                                -1& 
                              \end{cases}''')
        outp.shift(DOWN * 2)
        y2 = TexMobject('y', ' =')
        y2.next_to(outp, LEFT)
        seta = TexMobject('\\rightarrow')
        seta.next_to(net, buff=0)
        y1 = TexMobject('y')
        y1.next_to(seta)
        self.play(Write(net, run_time=4), ),
        self.play(Write(soma), Write(sign), Write(y1), Write(seta))
        #        self.add(soma, w0, w1, w2, wn, lb, lm, lc)
        self.wait()
        self.play(Indicate(soma))
        self.wait()
        self.play(Indicate(sign))
        self.wait()
        self.play(Indicate(y1))
        self.wait()
        self.play(ReplacementTransform(y1.copy(), y2[0]))
        self.play(Write(y2[1]), Write(outp))
        self.wait(2)
        self.play(Write(w0)), self.play(Write(w1))
        self.play(Write(w2)), self.play(Write(wn))
        self.wait(2)
        self.play(Indicate(w0))
        self.wait(2)
        self.play(FadeOut(y2), FadeOut(outp))
        self.wait()
        somatorio = TexMobject(r'y = sgn(\sum_{i=1}^{n} x_{i} w_{i} + w_{0})')
        somatorio.shift(DOWN * 2, RIGHT * 0.4)
        self.play(Write(somatorio))
        self.wait(2)
        dotp = TexMobject(r'y = sgn (\vec{x} \vec{w})')
        dotp.shift(DOWN * 2)
#        self.play(ReplacementTransform(somatorio, dotp))
        self.wait(2)


#        self.add(outp)

class NeuralNetworkMobject(VGroup):
    CONFIG = {
        "neuron_radius": 0.15,
        "neuron_to_neuron_buff": MED_SMALL_BUFF,
        "layer_to_layer_buff": LARGE_BUFF,
        "output_neuron_color": RED,
        "input_neuron_color": RED,
        "hidden_layer_neuron_color": RED,
        "neuron_stroke_width": 4,
        "neuron_fill_color": GREEN,
        "edge_color": GREY,
        "edge_stroke_width": 2,
        "edge_propogation_color": YELLOW,
        "edge_propogation_time": 1,
        "max_shown_neurons": 16,
        "brace_for_large_layers": True,
        "average_shown_activation_of_large_layer": True,
        "include_output_labels": False,
        "arrow": False,
        "arrow_tip_size": 0.1,
        "left_size": 1,
        "neuron_fill_opacity": 1
    }

    # Constructor with parameters of the neurons in a list
    def __init__(self, neural_network, *args, **kwargs):
        VGroup.__init__(self, *args, **kwargs)
        self.layer_sizes = neural_network
        self.add_neurons()
        self.add_edges()
        self.add_to_back(self.layers)

    # Helper method for constructor
    def add_neurons(self):
        layers = VGroup(*[
            self.get_layer(size, index)
            for index, size in enumerate(self.layer_sizes)
        ])
        layers.arrange_submobjects(RIGHT, buff=self.layer_to_layer_buff)
        self.layers = layers
        if self.include_output_labels:
            self.label_outputs_text()

    # Helper method for constructor
    def get_nn_fill_color(self, index):
        if index == -1 or index == len(self.layer_sizes) - 1:
            return self.output_neuron_color
        if index == 0:
            return self.input_neuron_color
        else:
            return self.hidden_layer_neuron_color

    # Helper method for constructor
    def get_layer(self, size, index=-1):
        layer = VGroup()
        n_neurons = size
        if n_neurons > self.max_shown_neurons:
            n_neurons = self.max_shown_neurons
        neurons = VGroup(*[
            Circle(
                radius=self.neuron_radius,
                stroke_color=self.get_nn_fill_color(index),
                stroke_width=self.neuron_stroke_width,
                fill_color=WHITE,
                fill_opacity=self.neuron_fill_opacity,
            )
            for x in range(n_neurons)
        ])
        neurons.arrange_submobjects(
            DOWN, buff=self.neuron_to_neuron_buff
        )
        for neuron in neurons:
            neuron.edges_in = VGroup()
            neuron.edges_out = VGroup()
        layer.neurons = neurons
        layer.add(neurons)

        if size > n_neurons:
            dots = TexMobject("\\vdots")
            dots.move_to(neurons)
            VGroup(*neurons[:len(neurons) // 2]).next_to(
                dots, UP, MED_SMALL_BUFF
            )
            VGroup(*neurons[len(neurons) // 2:]).next_to(
                dots, DOWN, MED_SMALL_BUFF
            )
            layer.dots = dots
            layer.add(dots)
            if self.brace_for_large_layers:
                brace = Brace(layer, LEFT)
                brace_label = brace.get_tex(str(size))
                layer.brace = brace
                layer.brace_label = brace_label
                layer.add(brace, brace_label)

        return layer

    # Helper method for constructor
    def add_edges(self):
        self.edge_groups = VGroup()
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            edge_group = VGroup()
            for n1, n2 in it.product(l1.neurons, l2.neurons):
                edge = self.get_edge(n1, n2)
                edge_group.add(edge)
                n1.edges_out.add(edge)
                n2.edges_in.add(edge)
            self.edge_groups.add(edge_group)
        self.add_to_back(self.edge_groups)

    # Helper method for constructor
    def get_edge(self, neuron1, neuron2):
        if self.arrow:
            return Arrow(
                neuron1.get_center(),
                neuron2.get_center(),
                buff=self.neuron_radius,
                stroke_color=self.edge_color,
                stroke_width=self.edge_stroke_width,
                tip_length=self.arrow_tip_size
            )
        return Line(
            neuron1.get_center(),
            neuron2.get_center(),
            buff=self.neuron_radius,
            stroke_color=self.edge_color,
            stroke_width=self.edge_stroke_width,
        )

    # Labels each input neuron with a char l or a LaTeX character
    def label_inputs(self, l):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[0].neurons):
            if n == 0:
                label = TexMobject('\\Phi (0)')

            elif n <= 2:
                label = TexMobject(f"{l}" + "{" + f"({n})" + "}")
            else:
                label = TexMobject(f"{l}_" + "{" + f"{'n'}" + "}")
            label.set_height(0.3 * neuron.get_height())
            label.move_to(neuron)
            label.set_color(BLACK)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels each output neuron with a char l or a LaTeX character
    def label_outputs(self, l):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = TexMobject(f"{l}_" + "{" + f"{'q'}" + "}")
            label.set_height(0.4 * neuron.get_height())
            label.move_to(neuron)
            label.set_color(BLACK)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels each neuron in the output layer with text according to an output list
    def label_outputs_text(self, outputs):
        self.output_labels = VGroup()
        for n, neuron in enumerate(self.layers[-1].neurons):
            label = TexMobject(outputs[n])
            label.set_height(0.75 * neuron.get_height())
            label.move_to(neuron)
            label.shift((neuron.get_width() + label.get_width() / 2) * RIGHT)
            self.output_labels.add(label)
        self.add(self.output_labels)

    # Labels the hidden layers with a char l or a LaTeX character
    def label_hidden_layers(self, l):
        self.output_labels = VGroup()
        for layer in self.layers[1:-1]:
            for n, neuron in enumerate(layer.neurons):
                label = TexMobject(f"{l}_{n + 1}")
                label.set_height(0.4 * neuron.get_height())
                label.move_to(neuron)
                self.output_labels.add(label)
        self.add(self.output_labels)

class Rel(Scene):
    CONFIG = {
        'camera_config': {'background_color': WHITE},

    }
    def construct(self):

        net = NeuralNetworkMobject([2,3,1])
        net.scale(4.8)
        self.add(net)
        net.label_inputs('\\Phi')
        net.label_outputs('x')