/*
 * Sample application to illustrate processing with ProGuard.
 *
 * Copyright (c) 2012-2019 Guardsquare NV
 */
package com.example;

import android.app.Activity;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.TextView;

/**
 * Sample activity that displays "Hello world!".
 */
public class HelloWorldActivity extends Activity
{
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);

        // Display the message.
        TextView view = new TextView(this);
        view.setText("Hello World");
        view.setGravity(Gravity.CENTER);
        setContentView(view);
    }
}
