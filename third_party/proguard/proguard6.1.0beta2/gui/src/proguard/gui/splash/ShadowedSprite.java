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
 * This Sprite adds a drop shadow to another Sprite.
 *
 * @author Eric Lafortune
 */
public class ShadowedSprite implements Sprite
{
    private final VariableInt    xOffset;
    private final VariableInt    yOffset;
    private final VariableDouble alpha;
    private final VariableInt    blur;
    private final Sprite         sprite;

    private float cachedAlpha = -1.0f;
    private Color cachedColor;


    /**
     * Creates a new ShadowedSprite.
     * @param xOffset the variable x-offset of the shadow, relative to the sprite itself.
     * @param yOffset the variable y-offset of the shadow, relative to the sprite itself.
     * @param alpha   the variable darkness of the shadow (between 0 and 1).
     * @param blur    the variable blur of the shadow (0 for sharp shadows, 1 or
     *                more for increasingly blurry shadows).
     * @param sprite  the Sprite to be painted with its shadow.
     */
    public ShadowedSprite(VariableInt    xOffset,
                          VariableInt    yOffset,
                          VariableDouble alpha,
                          VariableInt    blur,
                          Sprite         sprite)
    {
        this.xOffset = xOffset;
        this.yOffset = yOffset;
        this.alpha   = alpha;
        this.blur    = blur;
        this.sprite  = sprite;
    }


    // Implementation for Sprite.

    public void paint(Graphics graphics, long time)
    {
        double l = alpha.getDouble(time);
        int    b = blur.getInt(time) + 1;

        float a = 1.0f - (float)Math.pow(1.0 - l, 1.0/(b*b));
        if (a != cachedAlpha)
        {
            cachedAlpha = a;
            cachedColor = new Color(0f, 0f, 0f, a);
        }

        // Set up the shadow graphics.
        //OverrideGraphics2D g = new OverrideGraphics2D((Graphics2D)graphics);
        //g.setOverrideColor(cachedColor);

        // Set the shadow color.
        Color actualColor = graphics.getColor();
        graphics.setColor(cachedColor);

        int xo = xOffset.getInt(time) - b/2;
        int yo = yOffset.getInt(time) - b/2;

        // Draw the sprite's shadow.
        for (int x = 0; x < b; x++)
        {
            for (int y = 0; y < b; y++)
            {
                int xt = xo + x;
                int yt = yo + y;
                graphics.translate(xt, yt);
                sprite.paint(graphics, time);
                graphics.translate(-xt, -yt);
            }
        }

        // Restore the actual sprite color.
        graphics.setColor(actualColor);

        // Draw the sprite itself in the ordinary graphics.
        sprite.paint(graphics, time);
    }
}
