from manimlib.imports import *
class TestPath(VGroup):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name.__name__
        self.func = name
        self.title = Text(f"{self.name}", font="Monaco", stroke_width=0)
        self.title.set_height(0.24)
        self.line = Line(LEFT * 5, RIGHT * 5)
        self.dot = Dot(self.line.get_start())
        self.title.next_to(self.line, LEFT, buff=0.3)
        self.add(self.title, self.line, self.dot)


class RateFunctions(Scene):
    CONFIG = {
        "rate_functions": [
            smooth,
            linear,
            rush_into,
            rush_from,
            slow_into,
            double_smooth,
            there_and_back,
            running_start,
            wiggle,
            lingering,
            exponential_decay
        ],
        "rt": 3
    }

    def construct(self):
        time_ad = [*[Text("%d" % i, font="Arial", stroke_width=0).to_corner(UL) for i in range(1, 4)]][::-1]

        rf_group = VGroup(*[
            TestPath(rf)
            for rf in self.rate_functions
        ])
        for rf in rf_group:
            rf.title.set_color(TEAL)
            rf.line.set_color([RED, BLUE, YELLOW])
        rf_group.arrange(DOWN, aligned_edge=RIGHT)
        init_point = rf_group[0].line.get_start()
        init_point[0] = 0
        end_point = rf_group[-1].line.get_end()
        brace = Brace(rf_group[-1].line, DOWN, buff=0.5)
        brace_text = brace.get_text("\\tt run\\_time = %d" % self.rt).scale(0.8)
        end_point[0] = 0

        div_lines = VGroup()
        div_texts = VGroup()
        for i in range(11):
            proportion = i / 10
            text = TexMobject("\\frac{%s}{10}" % i)
            text.set_height(0.5)
            coord_proportion = rf_group[0].line.point_from_proportion(proportion)
            coord_proportion[1] = 0
            v_line = DashedLine(
                init_point + coord_proportion + UP * 0.5,
                end_point + coord_proportion + DOWN * 0.5,
                stroke_opacity=0.5
            )
            text.next_to(v_line, UP, buff=0.1)
            div_texts.add(text)
            div_lines.add(v_line)
        self.add(rf_group, div_lines, div_texts, brace, brace_text)
        for i in range(3):
            self.add(time_ad[i])
            self.wait()
            self.remove(time_ad[i])
        self.play(*[
            MoveAlongPath(vg.dot, vg.line, rate_func=vg.func)
            for vg in rf_group
        ],
                  run_time=self.rt
                  )
        self.wait(2)