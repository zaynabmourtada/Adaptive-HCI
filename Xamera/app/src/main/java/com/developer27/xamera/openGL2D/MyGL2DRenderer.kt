package com.developer27.xamera.openGL2D

import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class MyGL2DRenderer : GLSurfaceView.Renderer {

    private var vertexBuffer: FloatBuffer? = null
    private var vertexCount = 0

    // Matrices for a 2D orthographic projection
    private val projectionMatrix = FloatArray(16)
    private val viewMatrix = FloatArray(16)
    private val modelMatrix = FloatArray(16)
    private val mvpMatrix = FloatArray(16)

    // Shader program handles
    private var programHandle = 0
    private var positionHandle = 0
    private var colorHandle = 0
    private var mvpMatrixHandle = 0

    // Light blue color (approx #ADD8E6): R=0.6784, G=0.8470, B=0.9019, A=1.0
    // Feel free to tweak these values for a different shade.
    private val color = floatArrayOf(0.6784f, 0.8470f, 0.9019f, 1f)

    // Vertex Shader
    private val vertexShaderCode =
        """
        uniform mat4 uMVPMatrix;
        attribute vec4 vPosition;
        void main() {
            gl_Position = uMVPMatrix * vPosition;
        }
        """

    // Fragment Shader
    private val fragmentShaderCode =
        """
        precision mediump float;
        uniform vec4 vColor;
        void main() {
            gl_FragColor = vColor;
        }
        """

    override fun onSurfaceCreated(unused: GL10, config: EGLConfig) {
        // Clear the screen to black
        GLES20.glClearColor(0f, 0f, 0f, 1f)

        // Compile shaders
        val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderCode)
        val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderCode)

        // Link into an OpenGL program
        programHandle = GLES20.glCreateProgram().also {
            GLES20.glAttachShader(it, vertexShader)
            GLES20.glAttachShader(it, fragmentShader)
            GLES20.glLinkProgram(it)
        }

        // Simple camera at z=1, looking at origin
        Matrix.setLookAtM(
            viewMatrix, 0,
            0f, 0f, 1f,  // eye
            0f, 0f, 0f,  // center
            0f, 1f, 0f   // up
        )

        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
    }

    override fun onSurfaceChanged(unused: GL10, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)

        // Orthographic projection from -1..1 in both X and Y
        Matrix.orthoM(
            projectionMatrix, 0,
            -1f, 1f,   // left, right
            -1f, 1f,   // bottom, top
            0.1f, 10f  // near, far
        )
    }

    override fun onDrawFrame(unused: GL10) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

        val buf = vertexBuffer ?: return
        if (vertexCount == 0) return

        // Identity model matrix
        Matrix.setIdentityM(modelMatrix, 0)

        // MVP = Projection * View * Model
        Matrix.multiplyMM(mvpMatrix, 0, viewMatrix, 0, modelMatrix, 0)
        Matrix.multiplyMM(mvpMatrix, 0, projectionMatrix, 0, mvpMatrix, 0)

        // Use our shader program
        GLES20.glUseProgram(programHandle)

        // Get handle references
        positionHandle = GLES20.glGetAttribLocation(programHandle, "vPosition")
        colorHandle    = GLES20.glGetUniformLocation(programHandle, "vColor")
        mvpMatrixHandle= GLES20.glGetUniformLocation(programHandle, "uMVPMatrix")

        // Enable the vertex array
        GLES20.glEnableVertexAttribArray(positionHandle)
        GLES20.glVertexAttribPointer(
            positionHandle,
            3, // x,y,z
            GLES20.GL_FLOAT,
            false,
            0,
            buf
        )

        // Set the light blue color
        GLES20.glUniform4fv(colorHandle, 1, color, 0)

        // Pass MVP
        GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false, mvpMatrix, 0)

        // Attempt a much bolder line width
        // (Note: Some devices may ignore widths > 1.0)
        GLES20.glLineWidth(12f)

        // Draw lines
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, vertexCount)

        // Disable attribute array
        GLES20.glDisableVertexAttribArray(positionHandle)
    }

    /**
     * Called by MyGLSurfaceView/OpenGLActivity with the chosen character's line coords.
     */
    fun setPoints(points: List<Float>) {
        vertexCount = points.size / 3
        Log.d("MyGLRenderer", "setPoints(): Received $vertexCount vertices.")

        val bb = ByteBuffer.allocateDirect(points.size * 4).order(ByteOrder.nativeOrder())
        val fb = bb.asFloatBuffer()
        fb.put(points.toFloatArray())
        fb.position(0)

        vertexBuffer = fb
    }

    /**
     * Helper to compile a shader
     */
    private fun loadShader(type: Int, shaderCode: String): Int {
        return GLES20.glCreateShader(type).also { shader ->
            GLES20.glShaderSource(shader, shaderCode)
            GLES20.glCompileShader(shader)

            val status = IntArray(1)
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
            if (status[0] == 0) {
                val err = GLES20.glGetShaderInfoLog(shader)
                Log.e("MyGLRenderer", "Shader compile error: $err")
                GLES20.glDeleteShader(shader)
                throw RuntimeException("Shader compile error")
            }
        }
    }
}
