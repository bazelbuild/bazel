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
 * This Sprite represents an animated circle. It can optionally be filled.
 *
 * @author Eric Lafortune
 */
public class CircleSprite implements Sprite
{
    private final boolean     filled;
    private final VariableInt x;
    private final VariableInt y;
    private final VariableInt radius;


    /**
     * Creates a new CircleSprite.
     * @param filled specifies whether the rectangle should be filled.
     * @param x      the variable x-coordinate of the center of the circle.
     * @param y      the variable y-coordinate of the center of the circle.
     * @param radius the variable radius of the circle.
     */
    public CircleSprite(boolean     filled,
                        VariableInt x,
                        VariableInt y,
                        VariableInt radius)
    {
        this.filled = filled;
        this.x      = x;
        this.y      = y;
        this.radius = radius;
    }


    // Implementation for Sprite.

    public void paint(Graphics graphics, long time)
    {
        int xt = x.getInt(time);
        int yt = y.getInt(time);
        int r  = radius.getInt(time);

        if (filled)
        {
            graphics.fillOval(xt - r, yt - r, 2 * r, 2 * r);
        }
        else
        {
            graphics.drawOval(xt - r, yt - r, 2 * r, 2 * r);
        }
    }
}
