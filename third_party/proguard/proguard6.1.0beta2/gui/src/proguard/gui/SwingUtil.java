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
package proguard.gui;

import javax.swing.*;
import java.lang.reflect.InvocationTargetException;


/**
 * This utility class provides variants of the invocation method from the
 * <code>SwingUtilities</code> class.
 *
 * @see SwingUtilities
 * @author Eric Lafortune
 */
public class SwingUtil
{
    /**
     * Invokes the given Runnable in the AWT event dispatching thread,
     * and waits for it to finish. This method may be called from any thread,
     * including the event dispatching thread itself.
     * @see SwingUtilities#invokeAndWait(Runnable)
     * @param runnable the Runnable to be executed.
     */
    public static void invokeAndWait(Runnable runnable)
    throws InterruptedException, InvocationTargetException
    {
        try
        {
            if (SwingUtilities.isEventDispatchThread())
            {
                runnable.run();
            }
            else
            {
                SwingUtilities.invokeAndWait(runnable);
            }
        }
        catch (Exception ex)
        {
            // Ignore any exceptions.
        }
    }


    /**
     * Invokes the given Runnable in the AWT event dispatching thread, not
     * necessarily right away. This method may be called from any thread,
     * including the event dispatching thread itself.
     * @see SwingUtilities#invokeLater(Runnable)
     * @param runnable the Runnable to be executed.
     */
    public static void invokeLater(Runnable runnable)
    {
        if (SwingUtilities.isEventDispatchThread())
        {
            runnable.run();
        }
        else
        {
            SwingUtilities.invokeLater(runnable);
        }
    }
}
