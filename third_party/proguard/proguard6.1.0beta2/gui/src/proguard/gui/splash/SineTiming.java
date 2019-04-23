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
 * This Timing varies between 0 and 1, as a sine wave over time.
 *
 * @author Eric Lafortune
 */
public class SineTiming implements Timing
{
    private final long period;
    private final long phase;


    /**
     * Creates a new SineTiming.
     * @param period the time period for a full cycle.
     * @param phase  the phase of the cycle, which is added to the actual time.
     */
    public SineTiming(long period, long phase)
    {
        this.period = period;
        this.phase  = phase;
    }


    // Implementation for Timing.

    public double getTiming(long time)
    {
        // Compute the translated and scaled sine function.
        return 0.5 + 0.5 * Math.sin(2.0 * Math.PI * (time + phase) / period);
    }
}
