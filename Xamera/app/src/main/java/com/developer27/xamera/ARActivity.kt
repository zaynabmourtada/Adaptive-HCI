package com.developer27.xamera

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.developer27.xamera.databinding.ActivityArBinding
import com.google.ar.sceneform.ux.ArFragment

class ARActivity : AppCompatActivity() {

    private lateinit var binding: ActivityArBinding
    private lateinit var arFragment: ArFragment

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Inflate the layout (if using ViewBinding)
        binding = ActivityArBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Or if not using ViewBinding: setContentView(R.layout.activity_ar)

        // Obtain the ArFragment from the layout
        arFragment = supportFragmentManager
            .findFragmentById(R.id.arFragment) as ArFragment

        // -- Example: Add a plane tap listener if you want to place 3D objects on the plane
        // arFragment.setOnTapArPlaneListener { hitResult, plane, motionEvent ->
        //    // TODO: Place or load a 3D model, create an anchor, etc.
        // }
    }
}