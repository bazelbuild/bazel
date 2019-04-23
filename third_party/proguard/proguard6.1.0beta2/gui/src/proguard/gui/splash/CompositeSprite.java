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
 * This Sprite is the composition of a list of Sprite objects.
 *
 * @author Eric Lafortune
 */
public class CompositeSprite implements Sprite
{
    private final Sprite[] sprites;


    /**
     * Creates a new CompositeSprite.
     * @param sprites the array of Sprite objects to which the painting will
     *                be delegated, starting with the first element.
     */
    public CompositeSprite(Sprite[] sprites)
    {
        this.sprites = sprites;
    }


    // Implementation for Sprite.

    public void paint(Graphics graphics, long time)
    {
        // Draw the sprites.
        for (int index = 0; index < sprites.length; index++)
        {
            sprites[index].paint(graphics, time);
        }
    }
}
