package com.developer27.xamera.openGL2D

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

object charGenerator {

    /**
     * Main entry point: get the line-segment data for any ASCII letter A-Z or digit 0-9.
     * We'll store each shape as a series of line segments (pairs of points),
     * or build them dynamically using arc/circle helpers for curves.
     */
    fun getCoordinatesForChar(c: Char): FloatArray {
        // We'll uppercase letters so 'a' and 'A' share the same shape
        val ch = c.uppercaseChar()
        return when {
            ch in 'A'..'Z' -> getLetterShape(ch)
            ch in '0'..'9' -> getDigitShape(ch)
            else -> getFallbackShape()
        }
    }

    // Fallback shape if character unrecognized
    private fun getFallbackShape(): FloatArray {
        // A single horizontal line in the smaller bounding box
        return floatArrayOf(
            -0.3f, 0f, 0f,
            0.3f, 0f, 0f
        )
    }

    // -----------------------------------------------------------------
    // LETTERS

    private fun getLetterShape(ch: Char): FloatArray {
        return when (ch) {
            'A' -> buildLetterA()
            'B' -> buildLetterB()
            'C' -> buildArcLetterC()
            'D' -> buildLetterD()
            'E' -> buildLetterE()
            'F' -> buildLetterF()
            'G' -> buildLetterG()
            'H' -> buildLetterH()
            'I' -> buildLetterI()
            'J' -> buildLetterJ()
            'K' -> buildLetterK()
            'L' -> buildLetterL()
            'M' -> buildLetterM()
            'N' -> buildLetterN()
            'O' -> buildLetterO()
            'P' -> buildLetterP()
            'Q' -> buildLetterQ()
            'R' -> buildLetterR()
            'S' -> buildLetterS()
            'T' -> buildLetterT()
            'U' -> buildLetterU()
            'V' -> buildLetterV()
            'W' -> buildLetterW()
            'X' -> buildLetterX()
            'Y' -> buildLetterY()
            'Z' -> buildLetterZ()
            else -> getFallbackShape()
        }
    }

    // -----------------------------------------------------------------
    // DIGITS

    private fun getDigitShape(ch: Char): FloatArray {
        return when (ch) {
            '0' -> buildDigit0()  // Curvy 0
            '1' -> buildDigit1()
            '2' -> buildDigit2()
            '3' -> buildDigit3()
            '4' -> buildDigit4()
            '5' -> buildDigit5()
            '6' -> buildDigit6()
            '7' -> buildDigit7()
            '8' -> buildDigit8()
            '9' -> buildDigit9()
            else -> getFallbackShape()
        }
    }

    // -----------------------------------------------------------------
    // LETTER IMPLEMENTATIONS (examples)
    //
    // We'll show a few. The rest follow a similar approach: arcs + lines
    // to approximate a smaller, curvier design.

    private fun buildLetterA(): FloatArray {
        // We'll slightly shrink from [-0.3..0.3] in X, [-0.5..0.5] in Y
        // to keep a nice "A" shape:
        //   Left edge, Right edge, Crossbar
        return floatArrayOf(
            // Left edge
            -0.3f, -0.5f, 0f,   0f,   0.5f, 0f,
            // Right edge
            0f,   0.5f,  0f,   0.3f, -0.5f, 0f,
            // Crossbar
            -0.15f, 0f,  0f,   0.15f, 0f,  0f
        )
    }

    private fun buildLetterB(): FloatArray {
        // We'll approximate B with a small top arc and bottom arc
        // plus the vertical spine on the left.
        val spine = floatArrayOf(
            -0.3f,  0.5f, 0f,  -0.3f, -0.5f, 0f
        )
        val topArc  = buildArc(cx = -0.3f, cy = 0.25f, rx = 0.4f, ry = 0.25f,
            startDeg = -90f, endDeg = 90f, segments = 8)
        val bottomArc = buildArc(cx = -0.3f, cy = -0.25f, rx = 0.4f, ry = 0.25f,
            startDeg = -90f, endDeg = 90f, segments = 8)

        // Combine spine + topArc + bottomArc into one array
        return spine.combineWith(topArc).combineWith(bottomArc)
    }

    private fun buildArcLetterC(): FloatArray {
        // A smaller "C" from top to bottom using an arc
        // center ~ (0, 0), radius ~ 0.5 in Y, 0.3 in X
        // We'll approximate from about 45 deg to 315 deg to make a 'C' shape
        return buildArc(cx = 0f, cy = 0f,
            rx = 0.3f, ry = 0.5f,
            startDeg = 45f, endDeg = 315f,
            segments = 16)
    }

    private fun buildLetterD(): FloatArray {
        // Left spine + right arc
        val spine = floatArrayOf(
            -0.3f, 0.5f,0f,  -0.3f,-0.5f,0f
        )
        // Arc from top to bottom along the right
        val arcD = buildArc(cx = -0.3f, cy = 0f,
            rx = 0.3f, ry = 0.5f,
            startDeg = -90f, endDeg = 90f, // half-arc
            segments = 12)
        return spine.combineWith(arcD)
    }

    private fun buildLetterH(): FloatArray {
        // H: left vertical, right vertical, middle cross
        return floatArrayOf(
            // Left vertical
            -0.3f,0.5f,0f,   -0.3f,-0.5f,0f,
            // Right vertical
            0.3f,0.5f,0f,    0.3f,-0.5f,0f,
            // Middle cross
            -0.3f,0f,0f,      0.3f,0f,0f
        )
    }

    private fun buildLetterI(): FloatArray {
        // I: top horizontal + center vertical + bottom horizontal
        return floatArrayOf(
            // Top
            -0.1f, 0.5f,0f,   0.1f, 0.5f,0f,
            // Vertical
            0f, 0.5f,0f,     0f,-0.5f,0f,
            // Bottom
            -0.1f,-0.5f,0f,   0.1f,-0.5f,0f
        )
    }

    private fun buildLetterJ(): FloatArray {
        // J: top horizontal, right vertical, small bottom curve
        return floatArrayOf(
            // Top horizontal
            -0.1f,0.5f,0f,    0.1f,0.5f,0f,
            // Right vertical
            0.1f,0.5f,0f,    0.1f,-0.3f,0f,
            // Bottom curve (approx line from right to center)
            0.1f,-0.3f,0f,   -0.1f,-0.5f,0f
        )
    }

    private fun buildLetterK(): FloatArray {
        // K: left vertical + two diagonals
        return floatArrayOf(
            // Left vertical
            -0.3f,0.5f,0f,   -0.3f,-0.5f,0f,
            // Upper diagonal
            -0.3f,0f,  0f,    0.3f,0.5f,0f,
            // Lower diagonal
            -0.3f,0f,  0f,    0.3f,-0.5f,0f
        )
    }

    private fun buildLetterL(): FloatArray {
        // L: left vertical + bottom horizontal
        return floatArrayOf(
            -0.3f,0.5f,0f,   -0.3f,-0.5f,0f,
            -0.3f,-0.5f,0f,   0.2f,-0.5f,0f
        )
    }

    private fun buildLetterM(): FloatArray {
        // M: two verticals + two diagonals
        return floatArrayOf(
            // Left vertical
            -0.3f,-0.5f,0f,  -0.3f,0.5f,0f,
            // Diagonal up
            -0.3f,0.5f,0f,    0f,0f,0f,
            // Diagonal down
            0f,0f,0f,         0.3f,0.5f,0f,
            // Right vertical
            0.3f,0.5f,0f,     0.3f,-0.5f,0f
        )
    }

    private fun buildLetterN(): FloatArray {
        // N: left vertical, diagonal, right vertical
        return floatArrayOf(
            -0.3f,-0.5f,0f,  -0.3f,0.5f,0f,
            -0.3f,0.5f,0f,   0.3f,-0.5f,0f,
            0.3f,-0.5f,0f,   0.3f,0.5f,0f
        )
    }

    private fun buildLetterP(): FloatArray {
        // P: left vertical + top loop
        val spine = floatArrayOf(
            -0.3f,0.5f,0f,   -0.3f,-0.5f,0f
        )
        val topArc = buildArc(cx=-0.3f, cy=0.2f, rx=0.3f, ry=0.2f,
            startDeg=-90f, endDeg=90f, segments=8)
        return spine.combineWith(topArc)
    }

    private fun buildLetterQ(): FloatArray {
        // Q: circle + a small tail
        val circleO = buildCircle(cx=0f, cy=0f, rx=0.3f, ry=0.5f, segments=16)
        val tail    = floatArrayOf(
            0.1f,-0.2f,0f,  0.3f,-0.5f,0f
        )
        return circleO.combineWith(tail)
    }

    private fun buildLetterR(): FloatArray {
        // R: similar to P but with a diagonal
        val spine = floatArrayOf(
            -0.3f,0.5f,0f,   -0.3f,-0.5f,0f
        )
        val topArc = buildArc(cx=-0.3f, cy=0.2f, rx=0.3f, ry=0.2f,
            startDeg=-90f, endDeg=90f, segments=8)
        // diagonal
        val diag   = floatArrayOf(
            -0.3f,0f,0f,  0.3f,-0.5f,0f
        )
        return spine.combineWith(topArc).combineWith(diag)
    }

    private fun buildLetterS(): FloatArray {
        // We'll approximate an S with two arcs
        val topArc = buildArc(cx=0f, cy=0.2f, rx=0.3f, ry=0.3f,
            startDeg=180f, endDeg=340f, segments=8)
        val botArc = buildArc(cx=0f, cy=-0.2f, rx=0.3f, ry=0.3f,
            startDeg=0f, endDeg=160f, segments=8)
        return topArc.combineWith(botArc)
    }

    private fun buildLetterT(): FloatArray {
        // T: top horizontal + center vertical
        return floatArrayOf(
            // Top
            -0.3f,0.5f,0f,   0.3f,0.5f,0f,
            // Vertical
            0f, 0.5f,0f,     0f,-0.5f,0f
        )
    }

    private fun buildLetterU(): FloatArray {
        // U: left vertical, right vertical, bottom curve (approx line)
        return floatArrayOf(
            -0.3f,0.5f,0f,   -0.3f,-0.3f,0f,
            0.3f,0.5f,0f,    0.3f,-0.3f,0f,
            -0.3f,-0.3f,0f,   0.3f,-0.3f,0f
        )
    }

    private fun buildLetterV(): FloatArray {
        // V: two diagonals
        return floatArrayOf(
            -0.3f,0.5f,0f,   0f,-0.5f,0f,
            0f,-0.5f,0f,     0.3f,0.5f,0f
        )
    }

    private fun buildLetterW(): FloatArray {
        // W: four line segments
        return floatArrayOf(
            -0.3f,0.5f,0f,  -0.15f,-0.5f,0f,
            -0.15f,-0.5f,0f, 0.15f,-0.5f,0f,
            0.15f,-0.5f,0f,  0.3f,0.5f,0f
        )
    }

    private fun buildLetterX(): FloatArray {
        // X: two diagonals
        return floatArrayOf(
            -0.3f,0.5f,0f,   0.3f,-0.5f,0f,
            0.3f,0.5f,0f,  -0.3f,-0.5f,0f
        )
    }

    private fun buildLetterY(): FloatArray {
        // Y: top left diag, top right diag, vertical
        return floatArrayOf(
            -0.3f,0.5f,0f,   0f,0f,0f,
            0.3f,0.5f,0f,   0f,0f,0f,
            0f,0f,0f,       0f,-0.5f,0f
        )
    }

    private fun buildLetterZ(): FloatArray {
        // Z: top, diagonal, bottom
        return floatArrayOf(
            -0.3f,0.5f,0f,   0.3f,0.5f,0f,
            0.3f,0.5f,0f,  -0.3f,-0.5f,0f,
            -0.3f,-0.5f,0f,  0.3f,-0.5f,0f
        )
    }

    // ... [Other letters similarly built] ...

    private fun buildLetterO(): FloatArray {
        // "O" = a full circle
        return buildCircle(cx = 0f, cy = 0f, rx = 0.3f, ry = 0.5f, segments = 24)
    }

    // -----------------------------------------------------------------
    // DIGITS IMPLEMENTATIONS (examples)

    private fun buildDigit0(): FloatArray {
        // A vertical ellipse approximating "0"
        return buildCircle(cx = 0f, cy = 0f, rx = 0.3f, ry = 0.5f, segments = 24)
    }

    private fun buildDigit1(): FloatArray {
        // Slight top bar and foot bar
        return floatArrayOf(
            // Top bar
            -0.1f, 0.5f, 0f,   0.1f, 0.5f, 0f,
            // Vertical stem
            0f,   0.5f, 0f,   0f,  -0.5f, 0f,
            // Foot (bottom bar)
            -0.05f, -0.5f, 0f,  0.05f, -0.5f, 0f
        )
    }

    private fun buildDigit2(): FloatArray {
        // Top arc: from 180° to 360° (like a half-circle at the top)
        val topArc = buildArc(
            cx       = 0f,
            cy       = 0.4f,
            rx       = 0.2f,
            ry       = 0.12f,
            startDeg = 180f,
            endDeg   = 360f,
            segments = 24 // smoother than 8
        )
        // Diagonal from right edge of top arc down to middle-left
        val diag = floatArrayOf(
            0.2f,  0.4f, 0f,  -0.2f, 0f,   0f
        )
        // Bottom arc from (-0.2,0) to (0.3,-0.5)
        // We'll approximate a quarter-circle for a graceful curve
        val bottomArc = buildArc(
            cx       = 0.05f,
            cy       = -0.1f,
            rx       = 0.25f,
            ry       = 0.4f,
            startDeg = 180f,
            endDeg   = 270f,
            segments = 16
        )
        return topArc.combineWith(diag).combineWith(bottomArc)
    }

    private fun buildDigit3(): FloatArray {
        // Top arc from ~180°..360°
        val topArc = buildArc(
            cx       = 0.1f,
            cy       = 0.3f,
            rx       = 0.25f,
            ry       = 0.2f,
            startDeg = 180f,
            endDeg   = 360f,
            segments = 24
        )
        // Middle arc bridging from ~ y=0.1..y=-0.1
        val midArc = buildArc(
            cx       = 0.1f,
            cy       = 0.0f,
            rx       = 0.27f,
            ry       = 0.2f,
            startDeg = 190f,
            endDeg   = 350f,
            segments = 24
        )
        // Bottom arc from ~180°..360°
        val botArc = buildArc(
            cx       = 0.1f,
            cy       = -0.2f,
            rx       = 0.25f,
            ry       = 0.2f,
            startDeg = 180f,
            endDeg   = 360f,
            segments = 24
        )
        return topArc.combineWith(midArc).combineWith(botArc)
    }

    /**
     * Digit '4':
     * - A partial left vertical (from top to some middle Y)
     * - A diagonal from that middle-left to top-right
     * - A full right vertical from top-right down to bottom
     */
    /**
     * Digit '4':
     *  1) A short top bar from left to near the middle (x ~ 0.1).
     *  2) A full right vertical bar from top to bottom at x=0.1.
     *  3) A diagonal from (x=-0.3, y=0) to (x=0.1, y=0.5).
     *
     * The bounding box is roughly:
     *   X: [-0.3..0.1]
     *   Y: [-0.5..0.5]
     */
    private fun buildDigit4(): FloatArray {
        return floatArrayOf(
            // 1) Short top horizontal
            -0.3f,  0.5f, 0f,   0.1f,  0.5f, 0f,

            // 2) Right vertical
            0.1f,  0.5f, 0f,   0.1f, -0.5f, 0f,

            // 3) Diagonal from left-middle to top-right
            -0.3f,  0.0f, 0f,   0.1f,  0.5f, 0f
        )
    }


    private fun buildDigit5(): FloatArray {
        // Top horizontal
        val top = floatArrayOf(
            -0.3f,  0.5f, 0f,   0.2f,  0.5f, 0f
        )
        // Left vertical
        val left = floatArrayOf(
            -0.3f,  0.5f, 0f,  -0.3f,  0f,   0f
        )
        // Middle cross
        val mid = floatArrayOf(
            -0.3f,  0f,   0f,   0.2f,  0f,   0f
        )
        // Bottom arc from (0.2,0) to near (0.2,-0.5) with a curve
        val botArc = buildArc(
            cx       = -0.05f,
            cy       = -0.25f,
            rx       = 0.25f,
            ry       = 0.25f,
            startDeg = 0f,
            endDeg   = 90f,
            segments = 16
        )
        // line from arc end to bottom-left
        val botLeft = floatArrayOf(
            -0.3f, -0.5f, 0f,   -0.05f, -0.25f, 0f
        )
        return top.combineWith(left).combineWith(mid).combineWith(botArc).combineWith(botLeft)
    }


    private fun buildDigit6(): FloatArray {
        // Big arc from 30°..330°
        val bigArc = buildArc(
            cx       = 0f,
            cy       = 0f,
            rx       = 0.3f,
            ry       = 0.5f,
            startDeg = 30f,
            endDeg   = 330f,
            segments = 24
        )
        // Little cross line at middle
        val cross = floatArrayOf(
            0f,  0f, 0f,   0.2f,  0f, 0f
        )
        return bigArc.combineWith(cross)
    }


    private fun buildDigit7(): FloatArray {
        return floatArrayOf(
            // Top horizontal
            -0.3f,  0.5f, 0f,   0.3f,  0.5f, 0f,
            // Diagonal
            0.3f,  0.5f, 0f,  -0.1f, -0.5f, 0f
        )
    }


    private fun buildDigit8(): FloatArray {
        val topCircle = buildCircle(
            cx       = 0f,
            cy       = 0.2f,
            rx       = 0.2f,
            ry       = 0.25f,
            segments = 24
        )
        val botCircle = buildCircle(
            cx       = 0f,
            cy       = -0.2f,
            rx       = 0.2f,
            ry       = 0.25f,
            segments = 24
        )
        return topCircle.combineWith(botCircle)
    }

    private fun buildDigit9(): FloatArray {
        val topCircle = buildCircle(
            cx       = 0f,
            cy       = 0.2f,
            rx       = 0.2f,
            ry       = 0.25f,
            segments = 24
        )
        // Arc from ~270°..360° for the tail swirl
        val tailArc = buildArc(
            cx       = 0.1f,
            cy       = -0.2f,
            rx       = 0.2f,
            ry       = 0.3f,
            startDeg = 270f,
            endDeg   = 360f,
            segments = 16
        )
        return topCircle.combineWith(tailArc)
    }


    // -----------------------------------------------------------------
    // HELPER FUNCTIONS

    /**
     * Builds a full circle (ellipse) with 'segments' line segments.
     * If you want a perfect circle, keep rx = ry or choose different radii for ellipse.
     */
    private fun buildCircle(cx: Float, cy: Float, rx: Float, ry: Float, segments: Int): FloatArray {
        // We'll approximate the circle from angle=0..2π
        return buildArc(cx, cy, rx, ry, 0f, 360f, segments)
    }

    /**
     * Builds an arc from 'startDeg' to 'endDeg' using 'segments' line segments.
     * Positions: (cx + rx*cosθ, cy + ry*sinθ).
     */
    private fun buildArc(cx: Float, cy: Float,
                         rx: Float, ry: Float,
                         startDeg: Float, endDeg: Float,
                         segments: Int): FloatArray {

        // Convert to radians
        val startRad = startDeg * PI / 180.0
        val endRad   = endDeg   * PI / 180.0
        val arcPoints = mutableListOf<Float>()

        // Step in 'segments' increments
        val step = (endRad - startRad) / segments

        var angle = startRad
        var x0 = (cx + rx*cos(angle)).toFloat()
        var y0 = (cy + ry*sin(angle)).toFloat()

        for (i in 1..segments) {
            angle += step
            val x1 = (cx + rx*cos(angle)).toFloat()
            val y1 = (cy + ry*sin(angle)).toFloat()

            // Add line segment from (x0,y0) to (x1,y1)
            arcPoints.add(x0); arcPoints.add(y0); arcPoints.add(0f)
            arcPoints.add(x1); arcPoints.add(y1); arcPoints.add(0f)

            x0 = x1
            y0 = y1
        }
        return arcPoints.toFloatArray()
    }

    /**
     * Utility extension: combine two float arrays (each representing line segments).
     * You can then feed them to OpenGL as a single buffer.
     */
    private fun FloatArray.combineWith(other: FloatArray): FloatArray {
        val combined = FloatArray(this.size + other.size)
        System.arraycopy(this, 0, combined, 0, this.size)
        System.arraycopy(other, 0, combined, this.size, other.size)
        return combined
    }


    // The rest of the letter/digit builders follow a similar pattern
    // with arcs + lines in the smaller bounding box. You can refine
    // or tweak them for more polished shapes or different proportions.

    private fun buildLetterE(): FloatArray { /* ... */ return floatArrayOf(
        // Just an example
        -0.3f, 0.5f,0f, -0.3f, -0.5f,0f, // left vertical
        -0.3f, 0.5f,0f,  0.2f, 0.5f,0f,  // top
        -0.3f, 0f,0f,     0.1f, 0f,0f,   // middle
        -0.3f, -0.5f,0f,  0.2f,-0.5f,0f  // bottom
    )}

    private fun buildLetterF(): FloatArray { /* ... */ return floatArrayOf(
        // left vertical
        -0.3f, 0.5f,0f,  -0.3f,-0.5f,0f,
        // top
        -0.3f, 0.5f,0f,   0.2f, 0.5f,0f,
        // middle
        -0.3f, 0f,0f,     0.1f, 0f,0f
    )}

    private fun buildLetterG(): FloatArray { /* ...similar arcs ...*/ return buildArc(0f,0f, 0.3f,0.5f, 45f,315f,16).combineWith(
        // small line or arc inside
        floatArrayOf(
            0.0f,-0.1f,0f,  0.2f,-0.1f,0f
        )
    )}

}
