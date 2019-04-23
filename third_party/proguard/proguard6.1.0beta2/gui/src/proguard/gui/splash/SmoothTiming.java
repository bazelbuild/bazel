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
 * This Timing ramps up smoothly from 0 to 1 in a given time interval.
 *
 * @author Eric Lafortune
 */
public class SmoothTiming implements Timing
{
    private final long fromTime;
    private final long toTime;


    /**
     * Creates a new SmoothTiming.
     * @param fromTime the time at which the timing starts ramping up from 0.
     * @param toTime   the time at which the timing stops ramping up at 1.
     */
    public SmoothTiming(long fromTime, long toTime)
    {
        this.fromTime = fromTime;
        this.toTime   = toTime;
    }


    // Implementation for Timing.

    public double getTiming(long time)
    {
        if (time <= fromTime)
        {
            return 0.0;
        }

        if (time >= toTime)
        {
            return 1.0;
        }

        // Compute the linear interpolation.
        double timing = (double) (time - fromTime) / (double) (toTime - fromTime);

        // Smooth the interpolation at the ends.
        return timing * timing * (3.0 - 2.0 * timing);
    }
}
