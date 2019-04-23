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
 * This Sprite represents a text.
 *
 * @author Eric Lafortune
 */
public class TextSprite implements Sprite
{
    private final VariableString[] text;
    private final VariableInt      spacing;
    private final VariableInt      x;
    private final VariableInt      y;


    /**
     * Creates a new TextSprite containing a single line of text.
     * @param text  the variable text string.
     * @param x     the variable x-coordinate of the lower-left corner of the text.
     * @param y     the variable y-coordinate of the lower-left corner of the text.
     */
    public TextSprite(VariableString text,
                      VariableInt    x,
                      VariableInt    y)
    {
        this(new VariableString[] { text }, new ConstantInt(0), x, y);
    }


    /**
     * Creates a new TextSprite containing a multiple lines of text.
     * @param text    the variable text strings.
     * @param spacing the variable spacing between the lines of text.
     * @param x       the variable x-coordinate of the lower-left corner of the
     *                first line of text.
     * @param y       the variable y-coordinate of the lower-left corner of the
     *                first line of text.
     */
    public TextSprite(VariableString[] text,
                      VariableInt      spacing,
                      VariableInt      x,
                      VariableInt      y)
    {

        this.text    = text;
        this.spacing = spacing;
        this.x       = x;
        this.y       = y;
    }


    // Implementation for Sprite.

    public void paint(Graphics graphics, long time)
    {

        int xt = x.getInt(time);
        int yt = y.getInt(time);

        int spacingt = spacing.getInt(time);

        for (int index = 0; index < text.length; index++)
        {
            graphics.drawString(text[index].getString(time), xt, yt + index * spacingt);
        }
    }
}
