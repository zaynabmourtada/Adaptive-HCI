package com.developer27.xamera
import android.content.Context
import android.graphics.SurfaceTexture
import android.util.AttributeSet
import android.view.TextureView
class OpenGLTextureView(context: Context, attrs: AttributeSet?) : TextureView(context, attrs),
    TextureView.SurfaceTextureListener {
    private var renderer: OpenGLRenderer? = null
    private var renderThread: Thread? = null
    init {
        surfaceTextureListener = this
    }
    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
        renderer = OpenGLRenderer()
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