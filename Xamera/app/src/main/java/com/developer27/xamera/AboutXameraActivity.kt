package com.developer27.xamera

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.developer27.xamera.databinding.ActivityAboutXameraBinding

class AboutXameraActivity : AppCompatActivity() {

    private lateinit var binding: ActivityAboutXameraBinding
    private var umLogoClickCount = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAboutXameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // When the details container is clicked, open the URL in a browser.
        binding.detailsContainer.setOnClickListener {
            val url = "https://www.zhangxiao.me/"
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        }

        // When the UM logo is clicked 5 times, show a toast.
        binding.umLogo.setOnClickListener {
            umLogoClickCount++
            if (umLogoClickCount == 5) {
                Toast.makeText(this, "Wolverines beat Ohio!!", Toast.LENGTH_SHORT).show()
                umLogoClickCount = 0 // Reset the count after showing the toast
            } else{
                Toast.makeText(this, "You need ${5-umLogoClickCount} more clicks!", Toast.LENGTH_SHORT).show()
            }
        }
    }
}