/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.gui.splash;

import proguard.gui.SwingUtil;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.lang.reflect.InvocationTargetException;

/**
 * This JPanel renders an animated Sprite.
 *
 * @author Eric Lafortune
 */
public class SplashPanel extends JPanel
{
    private final MyAnimator  animator  = new MyAnimator();
    private final MyRepainter repainter = new MyRepainter();

    private final Sprite sprite;
    private final double sleepFactor;

    private long   startTime = Long.MAX_VALUE;
    private final long   stopTime;

    private volatile Thread animationThread;


    /**
     * Creates a new SplashPanel with the given Sprite, which will be animated
     * indefinitely.
     * @param sprite        the Sprite that will be animated.
     * @param processorLoad the fraction of processing time to be spend on
     *                      animating the Sprite (between 0 and 1).
     */
    public SplashPanel(Sprite sprite, double processorLoad)
    {
        this(sprite, processorLoad, (long)Integer.MAX_VALUE);
    }


    /**
     * Creates a new SplashPanel with the given Sprite, which will be animated
     * for a limited period of time.
     * @param sprite        the Sprite that will be animated.
     * @param processorLoad the fraction of processing time to be spend on
     *                      animating the Sprite (between 0 and 1).
     * @param stopTime      the number of milliseconds after which the
     *                      animation will be stopped automatically.
     */
    public SplashPanel(Sprite sprite, double processorLoad, long stopTime)
    {
        this.sprite      = sprite;
        this.sleepFactor = (1.0-processorLoad) / processorLoad;
        this.stopTime    = stopTime;

        // Restart the animation on a mouse click.
        addMouseListener(new MouseAdapter()
        {
            public void mouseClicked(MouseEvent e)
            {
                SplashPanel.this.start();
            }
        });
    }


    /**
     * Starts the animation.
     */
    public void start()
    {
        // Go to the beginning of the animation.
        startTime = System.currentTimeMillis();

        // Make sure we have an animation thread running.
        if (animationThread == null)
        {
            animationThread = new Thread(animator);
            animationThread.start();
        }
    }


    /**
     * Stops the animation.
     */
    public void stop()
    {
        // Go to the end of the animation.
        startTime = 0L;

        // Let the animation thread stop itself.
        animationThread = null;

        // Repaint the SplashPanel one last time.
        try
        {
            SwingUtil.invokeAndWait(repainter);
        }
        catch (InterruptedException ex)
        {
            // Nothing.
        }
        catch (InvocationTargetException ex)
        {
            // Nothing.
        }
    }


    // Implementation for JPanel.

    public void paintComponent(Graphics graphics)
    {
        super.paintComponent(graphics);

        sprite.paint(graphics, System.currentTimeMillis() - startTime);
    }


    /**
     * This Runnable makes sure its SplashPanel gets repainted regularly,
     * depending on the targeted processor load.
     */
    private class MyAnimator implements Runnable
    {
        public void run()
        {
            try
            {
                while (animationThread != null)
                {
                    // Check if we should stop the animation.
                    long time = System.currentTimeMillis();
                    if (time > startTime + stopTime)
                    {
                        animationThread = null;
                    }

                    // Do a repaint and time it.
                    SwingUtil.invokeAndWait(repainter);

                    // Sleep for a proportional while.
                    long repaintTime = System.currentTimeMillis() - time;
                    long sleepTime   = (long)(sleepFactor * repaintTime);
                    if (sleepTime < 10L)
                    {
                        sleepTime = 10L;
                    }

                    Thread.sleep(sleepTime);
                }
            }
            catch (InterruptedException ex)
            {
                // Nothing.
            }
            catch (InvocationTargetException ex)
            {
                // Nothing.
            }
        }
    }


    /**
     * This Runnable repaints its SplashPanel.
     */
    private class MyRepainter implements Runnable
    {
        public void run()
        {
            SplashPanel.this.repaint();
        }
    }


    /**
     * A main method for testing the splash panel.
     */
    public static void main(String[] args)
    {
        JFrame frame = new JFrame();
        frame.setTitle("Animation");
        frame.setSize(800, 600);
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        Dimension frameSize  = frame.getSize();
        frame.setLocation((screenSize.width - frameSize.width)   / 2,
                          (screenSize.height - frameSize.height) / 2);

        Sprite sprite =
            new ClipSprite(
            new ConstantColor(Color.white),
            new ConstantColor(Color.lightGray),
            new CircleSprite(true,
                             new LinearInt(200, 600, new SineTiming(2345L, 0L)),
                             new LinearInt(200, 400, new SineTiming(3210L, 0L)),
                             new ConstantInt(150)),
            new ColorSprite(new ConstantColor(Color.gray),
            new FontSprite(new ConstantFont(new Font("sansserif", Font.BOLD, 90)),
            new TextSprite(new ConstantString("ProGuard"),
                           new ConstantInt(200),
                           new ConstantInt(300)))));

        SplashPanel panel = new SplashPanel(sprite, 0.5);
        panel.setBackground(Color.white);

        frame.getContentPane().add(panel);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);

        panel.start();
    }
}
