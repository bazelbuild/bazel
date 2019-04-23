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
import java.awt.image.BufferedImage;

/**
 * This Sprite encapsulates another Sprite, which is then buffered in an Image.
 *
 * @author Eric Lafortune
 */
public class BufferedSprite implements Sprite
{
    private final int         bufferX;
    private final int         bufferY;
    private final Image       bufferImage;
    private final Color       backgroundColor;
    private final Sprite      sprite;
    private final VariableInt x;
    private final VariableInt y;

    private long cachedTime = -1;


    /**
     * Creates a new BufferedSprite with an ABGR image.
     * @param bufferX the x offset of the buffer image.
     * @param bufferY the y offset of the buffer image.
     * @param width   the width of the buffer image.
     * @param height  the height of the buffer image.
     * @param sprite  the Sprite that is painted in the buffer.
     * @param x       the variable x ordinate of the image buffer for painting.
     * @param y       the variable y ordinate of the image buffer for painting.
     *
     */
    public BufferedSprite(int         bufferX,
                          int         bufferY,
                          int         width,
                          int         height,
                          Sprite      sprite,
                          VariableInt x,
                          VariableInt y)
    {

        this(bufferX,
             bufferY,
             new BufferedImage(width, height, BufferedImage.TYPE_4BYTE_ABGR),
             null,
             sprite,
             x,
             y);
    }


    /**
     * Creates a new BufferedSprite with the given image.
     * @param bufferX         the x offset of the buffer image.
     * @param bufferY         the y offset of the buffer image.
     * @param bufferImage     the Image that is used for the buffering.
     * @param backgroundColor the background color that is used for the buffer.
     * @param sprite          the Sprite that is painted in the buffer.
     * @param x               the variable x ordinate of the image buffer for
     *                        painting.
     * @param y               the variable y ordinate of the image buffer for
     *                        painting.
     */
    public BufferedSprite(int         bufferX,
                          int         bufferY,
                          Image       bufferImage,
                          Color       backgroundColor,
                          Sprite      sprite,
                          VariableInt x,
                          VariableInt y)
    {
        this.bufferX         = bufferX;
        this.bufferY         = bufferY;
        this.bufferImage     = bufferImage;
        this.backgroundColor = backgroundColor;
        this.sprite          = sprite;
        this.x               = x;
        this.y               = y;
    }


   // Implementation for Sprite.

    public void paint(Graphics graphics, long time)
    {
        if (time != cachedTime)
        {
            Graphics bufferGraphics = bufferImage.getGraphics();

            // Clear the background.
            if (backgroundColor != null)
            {
                Graphics2D bufferGraphics2D = (Graphics2D)bufferGraphics;
                bufferGraphics2D.setComposite(AlphaComposite.Clear);
                bufferGraphics.fillRect(0, 0, bufferImage.getWidth(null), bufferImage.getHeight(null));
                bufferGraphics2D.setComposite(AlphaComposite.Src);
            }
            else
            {
                bufferGraphics.setColor(backgroundColor);
                bufferGraphics.fillRect(0, 0, bufferImage.getWidth(null), bufferImage.getHeight(null));
            }

            // Set up the buffer graphics.
            bufferGraphics.translate(-bufferX, -bufferY);
            bufferGraphics.setColor(graphics.getColor());
            bufferGraphics.setFont(graphics.getFont());

            // Draw the sprite.
            sprite.paint(bufferGraphics, time);

            bufferGraphics.dispose();

            cachedTime = time;
        }

        // Draw the buffer image.
        graphics.drawImage(bufferImage,
                           bufferX + x.getInt(time),
                           bufferY + y.getInt(time),
                           null);
    }
}
