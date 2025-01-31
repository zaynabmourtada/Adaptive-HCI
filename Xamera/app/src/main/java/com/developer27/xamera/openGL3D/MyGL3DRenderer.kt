package com.developer27.xamera.openGL3D

import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class MyGL3DRenderer : GLSurfaceView.Renderer {

    // Buffers
    private var touchBuffer: FloatBuffer? = null
    private var touchVertexCount = 0

    private var predefinedBuffer: FloatBuffer? = null
    private var predefinedVertexCount = 0

    private val color = floatArrayOf(1f, 0.3f, 0.3f, 1f)

    // Matrices
    private val modelMatrix = FloatArray(16)
    private val viewMatrix = FloatArray(16)
    private val projectionMatrix = FloatArray(16)
    private val mvpMatrix = FloatArray(16)

    private var programHandle = 0
    private var positionHandle = 0
    private var colorHandle = 0
    private var mvpMatrixHandle = 0

    private val vertexShaderCode = """
        uniform mat4 uMVPMatrix;
        attribute vec4 vPosition;
        void main() {
            gl_Position = uMVPMatrix * vPosition;
        }
    """

    private val fragmentShaderCode = """
        precision mediump float;
        uniform vec4 vColor;
        void main() {
            gl_FragColor = vColor;
        }
    """

    override fun onSurfaceCreated(unused: GL10, config: EGLConfig) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)

        val vs = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderCode)
        val fs = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderCode)

        programHandle = GLES20.glCreateProgram().also {
            GLES20.glAttachShader(it, vs)
            GLES20.glAttachShader(it, fs)
            GLES20.glLinkProgram(it)
        }

        // Camera at (0,0,6) looking at origin
        Matrix.setLookAtM(
            viewMatrix, 0,
            0f, 0f, 6f,   // eye
            0f, 0f, 0f,   // center
            0f, 1f, 0f    // up
        )

        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
    }

    override fun onSurfaceChanged(unused: GL10, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        val aspect = width.toFloat() / height
        Matrix.perspectiveM(
            projectionMatrix, 0,
            45f, aspect,
            0.1f, 100f
        )
    }

    override fun onDrawFrame(unused: GL10) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

        // Build a model matrix that does a slight rotation
        Matrix.setIdentityM(modelMatrix, 0)
        // E.g. rotate around X axis
        val time = (System.currentTimeMillis() % 10000L) / 10000f
        val angle = time * 360f
        Matrix.rotateM(modelMatrix, 0, angle, 1f, 0f, 0f)

        // Multiply: MVP = P * V * M
        Matrix.multiplyMM(mvpMatrix, 0, viewMatrix, 0, modelMatrix, 0)
        Matrix.multiplyMM(mvpMatrix, 0, projectionMatrix, 0, mvpMatrix, 0)

        GLES20.glUseProgram(programHandle)
        positionHandle = GLES20.glGetAttribLocation(programHandle, "vPosition")
        colorHandle    = GLES20.glGetUniformLocation(programHandle, "vColor")
        mvpMatrixHandle= GLES20.glGetUniformLocation(programHandle, "uMVPMatrix")

        // Draw pre-supplied data first
        if (predefinedVertexCount >= 2 && predefinedBuffer != null) {
            drawLineStrip(predefinedBuffer!!, predefinedVertexCount)
        }
        // or user touch data
        else if (touchVertexCount >= 2 && touchBuffer != null) {
            drawLineStrip(touchBuffer!!, touchVertexCount)
        }
    }

    private fun drawLineStrip(buffer: FloatBuffer, count: Int) {
        GLES20.glEnableVertexAttribArray(positionHandle)
        GLES20.glVertexAttribPointer(
            positionHandle,
            3,
            GLES20.GL_FLOAT,
            false,
            0,
            buffer
        )
        GLES20.glUniform4fv(colorHandle, 1, color, 0)
        GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false, mvpMatrix, 0)

        GLES20.glLineWidth(3f)
        GLES20.glDrawArrays(GLES20.GL_LINE_STRIP, 0, count)

        GLES20.glDisableVertexAttribArray(positionHandle)
    }

    /**
     * Called if user is drawing via touch in MyGL3DSurfaceView
     */
    fun setUserPoints(points: List<Float>) {
        touchVertexCount = points.size / 3
        val bb = ByteBuffer.allocateDirect(points.size * 4).order(ByteOrder.nativeOrder())
        val fb = bb.asFloatBuffer()
        fb.put(points.toFloatArray())
        fb.position(0)
        touchBuffer = fb
    }

    fun setPredefinedPoints(points: FloatArray) {
        predefinedVertexCount = points.size / 3
        val bb = ByteBuffer.allocateDirect(points.size * 4).order(ByteOrder.nativeOrder())
        val fb = bb.asFloatBuffer()
        fb.put(points)
        fb.position(0)
        predefinedBuffer = fb
    }

    private fun loadShader(type: Int, code: String): Int {
        return GLES20.glCreateShader(type).also { shader ->
            GLES20.glShaderSource(shader, code)
            GLES20.glCompileShader(shader)
            val status = IntArray(1)
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
            if (status[0] == 0) {
                val info = GLES20.glGetShaderInfoLog(shader)
                GLES20.glDeleteShader(shader)
                throw RuntimeException("Shader compile error: $info")
            }
        }
    }
}