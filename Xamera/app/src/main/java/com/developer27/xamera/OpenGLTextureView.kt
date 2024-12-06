package com.developer27.xamera

import android.content.Context
import android.graphics.SurfaceTexture
import android.util.AttributeSet
import android.view.TextureView

class OpenGLTextureView(
    context: Context,
    attrs: AttributeSet?
) : TextureView(context, attrs), TextureView.SurfaceTextureListener {

    private var renderer: OpenGLRenderer? = null
    private var renderThread: Thread? = null
    private var videoProcessor: VideoProcessor? = null // Add this field to hold VideoProcessor

    init {
        surfaceTextureListener = this
    }

    // Method to set the VideoProcessor
    fun setVideoProcessor(videoProcessor: VideoProcessor) {
        this.videoProcessor = videoProcessor
    }

    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
        if (videoProcessor == null) {
            throw IllegalStateException("VideoProcessor must be set before the surface is available.")
        }

        // Create a stable local reference
        val vp = videoProcessor!!

        // Now use 'vp' instead of 'videoProcessor'
        renderer = OpenGLRenderer(vp)
        renderThread = Thread { renderer?.start(surface, width, height) }
        renderThread?.start()
    }

    override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
        renderer?.onSurfaceChanged(width, height)
    }

    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
        renderer?.stop()
        renderThread?.join()
        renderer = null
        return true
    }

    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}
}