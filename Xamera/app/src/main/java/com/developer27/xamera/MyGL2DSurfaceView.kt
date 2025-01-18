package com.developer27.xamera

import android.content.Context
import android.opengl.GLSurfaceView

class MyGL2DSurfaceView(context: Context) : GLSurfaceView(context) {

    private val renderer: MyGL2DRenderer

    init {
        // Use OpenGL ES 2.0
        setEGLContextClientVersion(2)

        renderer = MyGL2DRenderer()
        setRenderer(renderer)

        // Draw only on request
        renderMode = RENDERMODE_WHEN_DIRTY
    }

    /**
     * Called by OpenGLActivity when the user picks a char's coordinates.
     */
    fun setPoints(points: List<Float>) {
        renderer.setPoints(points)
        requestRender()
    }
}
