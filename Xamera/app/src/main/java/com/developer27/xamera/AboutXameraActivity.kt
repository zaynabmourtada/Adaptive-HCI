package com.developer27.xamera

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.developer27.xamera.databinding.ActivityAboutXameraBinding

class AboutXameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityAboutXameraBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAboutXameraBinding.inflate(layoutInflater)
        setContentView(binding.root)
    }
}