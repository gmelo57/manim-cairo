from manimlib.imports import *


class MmodNTracker(Scene):
    CONFIG = {
        "number_of_lines": 100,
        "gradient_colors": [RED, YELLOW, BLUE],
        "end_value": 2,
        "total_time": 4,
    }

    def construct(self):
        circle = Circle().set_height(FRAME_HEIGHT * 0.9)
        mod_tracker = ValueTracker(0)
        lines = self.get_m_mod_n_objects(circle, mod_tracker.get_value())
        lines.add_updater(
            lambda mob: mob.become(
                self.get_m_mod_n_objects(circle, mod_tracker.get_value())
            )
        )
        self.add(circle, lines)
        self.wait(3)
        self.play(
            mod_tracker.set_value, self.end_value,
            rate_func=linear,
            run_time=self.total_time
        )
        self.wait(3)

    def get_m_mod_n_objects(self, circle, x, y=None):
        if y == None:
            y = self.number_of_lines
        lines = VGroup()
        for i in range(y):
            start_point = circle.point_from_proportion((i % y) / y)
            end_point = circle.point_from_proportion(((i * x) % y) / y)
            line = Line(start_point, end_point).set_stroke(width=1)
            lines.add(line)
        lines.set_color_by_gradient(*self.gradient_colors)
        return lines


class SimpleMmodN(Scene):
    def construct(self):
        circle, lines = self.get_m_mod_n_objects(2, 60)
        lines.rotate(-PI / 2)
        self.play(ShowCreation(VGroup(circle, lines), run_time=4))
        #        self.play(ShowCreationThenDestruction(lines, color=PINK, run_time=4))

        lines_c = lines.copy()
        lines_c.set_color(PINK)
        lines_c.set_stroke(width=3)
        #        self.play(LaggedStartMap(ShowCreationThenDestruction,lines_c))
        self.play(
            self.LaggedStartShowCrationThenDestructionLines(lines_c)
        )

    def LaggedStartShowCrationThenDestructionLines(self, lines):
        return LaggedStartMap(
            ShowCreationThenDestruction, lines,

        )

    def get_m_mod_n_objects(self, x, y):
        circle = Circle().set_height(FRAME_HEIGHT)
        circle.scale(0.85)
        lines = VGroup()
        for i in range(y):
            start_point = circle.point_from_proportion((i % y) / y)
            end_point = circle.point_from_proportion(((i * x) % y) / y)
            line = Line(start_point, end_point).set_stroke(width=1)
            lines.add(line)
        return [circle, lines]


old_version = True


class MmodN(SimpleMmodN):
    def construct(self):
        circle = Circle().set_height(FRAME_HEIGHT)
        circle.scale(0.85)
        circle.to_edge(RIGHT, buff=1)
        self.play(ShowCreation(circle))
        for x, y in [(2, 100), (3, 60), (4, 60), (5, 70)]:
            self.Example3b1b(self.get_m_mod_n_objects(x, y), x, y)
        self.play(FadeOut(circle))

    def Example3b1b(self, obj, x, y):
        circle, lines = obj
        lines.set_stroke(width=1)
        label = TexMobject(f"f({x},{y})").scale(2.5).to_edge(LEFT, buff=1)
        VGroup(circle, lines).to_edge(RIGHT, buff=1)
        self.play(
            Write(label),
            self.LaggedStartLines(lines)
        )
        self.wait()
        lines_c = lines.copy()
        lines_c.set_color(PINK)
        lines_c.set_stroke(width=3)
        self.play(
            self.LaggedStartShowCrationThenDestructionLines(lines_c)
        )
        self.wait()
        self.play(FadeOut(lines), Write(label, rate_func=lambda t: smooth(1 - t)))

    def LaggedStartLines(self, lines):
        return LaggedStartMap(
            ShowCreation,lines,
                         run_time=4
        )

    def LaggedStartShowCrationThenDestructionLines(self, lines):
        return LaggedStartMap(
            ShowCreationThenDestruction,lines,
                                        run_time=6
        )
