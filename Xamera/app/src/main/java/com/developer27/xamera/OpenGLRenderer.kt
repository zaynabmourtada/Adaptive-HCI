package com.developer27.xamera

import android.graphics.SurfaceTexture
import android.opengl.*
import android.util.Log
import android.widget.Toast
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

// TODO - Alan Raj <Due December 6th 2024>: Implement the isolated code you have created for OpenGL here.
// Ensure synchronization with Soham's Video Processing Code.
// Alan successfully setup the project.

/**
 * The OpenGLRenderer sets up an EGL context, manages an OpenGL ES 2.0 rendering pipeline,
 * and renders a rotating, scaling triangle. It retrieves data from the VideoProcessor,
 * although that data is not currently used in the rendering logic.
 */
class OpenGLRenderer(private val videoProcessor: VideoProcessor) {
    // EGL context-related members
    private var eglDisplay: EGLDisplay? = null
    private var eglContext: EGLContext? = null
    private var eglSurface: EGLSurface? = null
    // Rendering control flag
    private var running = false
    // The triangle to be drawn
    private var triangle: Triangle? = null
    // Transformation matrices for the rendering pipeline
    private val mvpMatrix = FloatArray(16)
    private val projectionMatrix = FloatArray(16)
    private val viewMatrix = FloatArray(16)
    private val modelMatrix = FloatArray(16)
    // Rotation angle for the animation
    private var angle = 0.0f
    /**
     * Begins the rendering process by initializing EGL and starting the render loop.
     */
    fun start(surfaceTexture: SurfaceTexture, width: Int, height: Int) {
        setupEGL(surfaceTexture)
        onSurfaceCreated()
        onSurfaceChanged(width, height)
        renderLoop()
        release()
    }
    /**
     * Sets up the EGL environment: display, context, and surface.
     */
    private fun setupEGL(surfaceTexture: SurfaceTexture) {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            throw RuntimeException("Unable to get EGL14 display")
        }
        val version = IntArray(2)
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            throw RuntimeException("Unable to initialize EGL14")
        }
        // Attributes for choosing a suitable EGL configuration
        val eglConfigAttributes = intArrayOf(
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_NONE
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        if (!EGL14.eglChooseConfig(
                eglDisplay, eglConfigAttributes, 0,
                configs, 0, configs.size, numConfigs, 0
            )
        ) {
            throw RuntimeException("Unable to choose EGL config")
        }
        // Attributes for creating an OpenGL ES 2.0 context
        val eglContextAttributes = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
        eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, eglContextAttributes, 0)
        // Attributes for creating a window surface
        val eglSurfaceAttributes = intArrayOf(EGL14.EGL_NONE)
        eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, configs[0], surfaceTexture, eglSurfaceAttributes, 0)

        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            throw RuntimeException("Failed to make EGL context current")
        }
    }
    /**
     * Main render loop that runs until 'stop()' is called. Clears the screen and redraws.
     */
    private fun renderLoop() {
        running = true
        while (running) {
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
            onDrawFrame()
            EGL14.eglSwapBuffers(eglDisplay, eglSurface)
        }
    }
    /**
     * Called once the surface is created. Sets the initial OpenGL state.
     */
    private fun onSurfaceCreated() {
        GLES20.glClearColor(0f, 0f, 0f, 0f)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        triangle = Triangle()
    }
    /**
     * Called when the surface size changes (e.g., device rotation).
     * Adjusts the viewport and projection matrix.
     */
    fun onSurfaceChanged(width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        val aspectRatio = width.toFloat() / height
        Matrix.frustumM(projectionMatrix, 0, -aspectRatio, aspectRatio, -1f, 1f, 3f, 7f)
    }
    /**
     * Called each frame to update transformations and draw the scene.
     */
    private fun onDrawFrame() {
        // Data retrieval from videoProcessor (currently unused in drawing logic)
        val preFilterData = videoProcessor.retrievePreFilter4Ddata()
        //val postFilterData = videoProcessor.retrievePostFilter4Ddata()

        if (preFilterData.isNotEmpty()) {
            val latestFrame = preFilterData.last() // Access the last element
            logDebug("OpenGL - Raw Data: | Frame(T)=${latestFrame.frameCount} | X=${latestFrame.x} | Y=${latestFrame.y} | Area=${latestFrame.area}")
        }
        // The preFilterData and postFilterData are obtained but not used.

        // Set up the camera view
        Matrix.setLookAtM(viewMatrix, 0,
            0f, 0f, -5f,  // Eye position
            0f, 0f, 0f,   // Look at center
            0f, 1f, 0f    // Up vector
        )
        // Apply rotation and scaling to the model matrix
        Matrix.setIdentityM(modelMatrix, 0)
        angle += 1.0f
        Matrix.rotateM(modelMatrix, 0, angle, 0f, 0f, 1f)
        val scaleFactor = 0.5f + 0.5f * kotlin.math.abs(kotlin.math.sin(Math.toRadians(angle.toDouble()))).toFloat()
        Matrix.scaleM(modelMatrix, 0, scaleFactor, scaleFactor, scaleFactor)
        // Combine the view and model matrices, then apply the projection
        Matrix.multiplyMM(mvpMatrix, 0, viewMatrix, 0, modelMatrix, 0)
        Matrix.multiplyMM(mvpMatrix, 0, projectionMatrix, 0, mvpMatrix, 0)
        // Draw the triangle with the final MVP matrix
        triangle?.draw(mvpMatrix)
    }
    /**
     * Stops the render loop.
     */
    fun stop() {
        running = false
    }
    /**
     * Cleans up and releases all EGL resources.
     */
    private fun release() {
        if (eglDisplay != null && eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(
                eglDisplay, EGL14.EGL_NO_SURFACE,
                EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT
            )
            EGL14.eglDestroySurface(eglDisplay, eglSurface)
            EGL14.eglDestroyContext(eglDisplay, eglContext)
            EGL14.eglTerminate(eglDisplay)
        }

        eglDisplay = null
        eglContext = null
        eglSurface = null
    }
    private fun logDebug(message: String) {
        if (Settings.Debug.enableLogging) Log.d("VideoProcessor", message)
    }
}

/**
 * The Triangle class sets up a simple vertex and fragment shader,
 * compiles them into a program, and draws a single triangle using that program.
 */
class Triangle {
    companion object {
        // Defines a simple triangle in normalized device coordinates
        private val vertices = floatArrayOf(
            0.0f,  0.5f, 0.0f,  // Top vertex
            -0.5f, -0.5f, 0.0f, // Bottom-left vertex
            0.5f, -0.5f, 0.0f   // Bottom-right vertex
        )
        private const val COORDS_PER_VERTEX = 3
        private val color = floatArrayOf(0.0f, 1.0f, 0.0f, 1.0f) // Green color
    }

    private val vertexBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(vertices.size * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer().apply {
            put(vertices)
            position(0)
        }
    private var shaderProgram: Int
    init {
        val vertexShaderCode = """
            uniform mat4 uMVPMatrix;
            attribute vec4 vPosition;
            void main() {
                gl_Position = uMVPMatrix * vPosition;
            }
        """.trimIndent()

        val fragmentShaderCode = """
            precision mediump float;
            uniform vec4 vColor;
            void main() {
                gl_FragColor = vColor;
            }
        """.trimIndent()

        val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderCode)
        val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderCode)

        shaderProgram = GLES20.glCreateProgram().also {
            GLES20.glAttachShader(it, vertexShader)
            GLES20.glAttachShader(it, fragmentShader)
            GLES20.glLinkProgram(it)
        }
    }
    /**
     * Draws the triangle using the provided MVP matrix.
     */
    fun draw(mvpMatrix: FloatArray) {
        GLES20.glUseProgram(shaderProgram)

        val positionHandle = GLES20.glGetAttribLocation(shaderProgram, "vPosition").also {
            GLES20.glEnableVertexAttribArray(it)
            GLES20.glVertexAttribPointer(
                it,
                COORDS_PER_VERTEX,
                GLES20.GL_FLOAT,
                false,
                COORDS_PER_VERTEX * 4,
                vertexBuffer
            )
        }

        GLES20.glGetUniformLocation(shaderProgram, "vColor").also {
            GLES20.glUniform4fv(it, 1, color, 0)
        }

        GLES20.glGetUniformLocation(shaderProgram, "uMVPMatrix").also {
            GLES20.glUniformMatrix4fv(it, 1, false, mvpMatrix, 0)
        }

        GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, vertices.size / COORDS_PER_VERTEX)
        GLES20.glDisableVertexAttribArray(positionHandle)
    }
    /**
     * Compiles an OpenGL shader from the given source code.
     */
    private fun loadShader(type: Int, shaderCode: String): Int {
        return GLES20.glCreateShader(type).also { shader ->
            GLES20.glShaderSource(shader, shaderCode)
            GLES20.glCompileShader(shader)
        }
    }
}