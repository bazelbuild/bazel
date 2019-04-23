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

/**
 * This VariableColor varies linearly with respect to its Timing.
 *
 * @author Eric Lafortune
 */
public class LinearInt implements VariableInt
{
    private final int    fromValue;
    private final int    toValue;
    private final Timing timing;


    /**
     * Creates a new LinearInt.
     * @param fromValue the value that corresponds to a timing of 0.
     * @param toValue   the value that corresponds to a timing of 1.
     * @param timing    the applied timing.
     */
    public LinearInt(int fromValue, int toValue, Timing timing)
    {
        this.fromValue = fromValue;
        this.toValue   = toValue;
        this.timing    = timing;
    }


    // Implementation for VariableInt.

    public int getInt(long time)
    {
        return (int) (fromValue + timing.getTiming(time) * (toValue - fromValue));
    }
}
