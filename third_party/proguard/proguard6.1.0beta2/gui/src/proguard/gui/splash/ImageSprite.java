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
 * This Sprite represents an animated image.
 *
 * @author Eric Lafortune
 */
public class ImageSprite implements Sprite
{
    private final Image          image;
    private final VariableInt    x;
    private final VariableInt    y;
    private final VariableDouble scaleX;
    private final VariableDouble scaleY;


    /**
     * Creates a new ImageSprite.
     * @param image  the Image to be painted.
     * @param x      the variable x-coordinate of the upper-left corner of the image.
     * @param y      the variable y-coordinate of the upper-left corner of the image.
     * @param scaleX the variable x-scale of the image.
     * @param scaleY the variable y-scale of the image.
     */
    public ImageSprite(Image          image,
                       VariableInt    x,
                       VariableInt    y,
                       VariableDouble scaleX,
                       VariableDouble scaleY)
    {
        this.image  = image;
        this.x      = x;
        this.y      = y;
        this.scaleX = scaleX;
        this.scaleY = scaleY;
    }


    // Implementation for Sprite.

    public void paint(Graphics graphics, long time)
    {
        int xt = x.getInt(time);
        int yt = y.getInt(time);

        double scale_x = scaleX.getDouble(time);
        double scale_y = scaleY.getDouble(time);

        int width  = (int)(image.getWidth(null)  * scale_x);
        int height = (int)(image.getHeight(null) * scale_y);

        graphics.drawImage(image, xt, yt, width, height, null);
    }
}
