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
 * This Sprite colors another given sprite.
 *
 * @author Eric Lafortune
 */
public class ColorSprite implements Sprite
{
    private final VariableColor color;
    private final Sprite        sprite;


    /**
     * Creates a new ColorSprite.
     * @param color  the variable color of the given sprite.
     * @param sprite the sprite that will be colored and painted.
     */
    public ColorSprite(VariableColor color,
                       Sprite        sprite)
    {
        this.color  = color;
        this.sprite = sprite;
    }


    // Implementation for Sprite.

    public void paint(Graphics graphics, long time)
    {
        // Save the old color.
        Color oldColor = graphics.getColor();

        // Set the new color.
        graphics.setColor(color.getColor(time));

        // Paint the actual sprite.
        sprite.paint(graphics, time);

        // Restore the old color.
        graphics.setColor(oldColor);
    }
}
