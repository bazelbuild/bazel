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

import java.awt.*;

/**
 * This Sprite displays another Sprite in a given time interval.
 * The time of the encapsulated Sprite is shifted by the start time.
 *
 * @author Eric Lafortune
 */
public class TimeSwitchSprite implements Sprite
{
    private final long   onTime;
    private final long offTime;
    private final Sprite sprite;


    /**
     * Creates a new TimeSwitchSprite for displaying a given Sprite starting at
     * a given time.
     * @param onTime the start time.
     * @param sprite the toggled Sprite.
     */
    public TimeSwitchSprite(long onTime, Sprite sprite)
    {
        this(onTime, 0L, sprite);
    }


    /**
     * Creates a new TimeSwitchSprite for displaying a given Sprite in a given
     * time interval.
     * @param onTime  the start time.
     * @param offTime the stop time.
     * @param sprite  the toggled Sprite.
     */
    public TimeSwitchSprite(long onTime, long offTime, Sprite sprite)
    {
        this.onTime  = onTime;
        this.offTime = offTime;
        this.sprite  = sprite;
    }


    // Implementation for Sprite.

    public void paint(Graphics graphics, long time)
    {
        if (time >= onTime && (offTime <= 0 || time <= offTime))
        {
            sprite.paint(graphics, time - onTime);
        }

    }
}
