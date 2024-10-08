package com.example.proyectodsm

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.lifecycle.ViewModelProvider
import com.example.proyectodsm.ui.AppModel
import com.example.proyectodsm.ui.AppViewModel

class MainActivity : ComponentActivity() {
    private lateinit var viewModel: AppViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewModel = ViewModelProvider(this).get(AppViewModel::class.java)

        AppModel.load(this)

        setContent {
            MainScreen(viewModel)
        }
    }
}
