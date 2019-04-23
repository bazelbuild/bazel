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
 * This Sprite encapsulates another Sprite, which is clipped by a clip Sprite.
 *
 * @author Eric Lafortune
 */
public class ClipSprite implements Sprite
{
    private final VariableColor insideClipColor;
    private final VariableColor outsideClipColor;
    private final Sprite        clipSprite;
    private final Sprite        sprite;


    /**
     * Creates a new ClipSprite.
     * @param insideClipColor  the background color inside the clip sprite.
     * @param outsideClipColor the background color outside the clip sprite.
     * @param clipSprite       the clip Sprite.
     * @param sprite           the clipped Sprite.
     */
    public ClipSprite(VariableColor insideClipColor,
                      VariableColor outsideClipColor,
                      Sprite        clipSprite,
                      Sprite        sprite)
    {
        this.insideClipColor  = insideClipColor;
        this.outsideClipColor = outsideClipColor;
        this.clipSprite       = clipSprite;
        this.sprite           = sprite;
    }


    // Implementation for Sprite.

    public void paint(Graphics graphics, long time)
    {
        // Clear the background.
        Color outsideColor = outsideClipColor.getColor(time);
        Rectangle clip = graphics.getClipBounds();
        graphics.setPaintMode();
        graphics.setColor(outsideColor);
        graphics.fillRect(0, 0, clip.width, clip.height);

        // Draw the sprite in XOR mode.
        OverrideGraphics2D g = new OverrideGraphics2D((Graphics2D)graphics);
        Color insideColor = insideClipColor.getColor(time);
        g.setOverrideXORMode(insideColor);
        sprite.paint(g, time);
        g.setOverrideXORMode(null);

        // Clear the clip area.
        g.setOverrideColor(insideColor);
        clipSprite.paint(g, time);
        g.setOverrideColor(null);

        // Draw the sprite in XOR mode.
        g.setOverrideXORMode(insideColor);
        sprite.paint(g, time);
        g.setOverrideXORMode(null);
    }
}
