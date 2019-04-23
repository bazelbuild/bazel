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
import java.awt.*;


/**
 * This <code>Runnable</code> can show a message dialog.
 *
 * @author Eric Lafortune
 */
final class MessageDialogRunnable implements Runnable
{
    private final Component parentComponent;
    private final Object    message;
    private final String    title;
    private final int       messageType;


    /**
     * Creates a new MessageDialogRunnable object.
     * @see JOptionPane#showMessageDialog(Component, Object, String, int)
     */
    public static void showMessageDialog(Component parentComponent,
                                         Object    message,
                                         String    title,
                                         int       messageType)
    {
        try
        {
            SwingUtil.invokeAndWait(new MessageDialogRunnable(parentComponent,
                                                              message,
                                                              title,
                                                              messageType));
        }
        catch (Exception e)
        {
            // Nothing.
        }
    }


    /**
     * Creates a new MessageDialogRunnable object.
     * @see JOptionPane#showMessageDialog(Component, Object, String, int)
     */
    public MessageDialogRunnable(Component parentComponent,
                                 Object    message,
                                 String    title,
                                 int       messageType)
    {
        this.parentComponent = parentComponent;
        this.message         = message;
        this.title           = title;
        this.messageType     = messageType;
    }



    // Implementation for Runnable.

    public void run()
    {
        JOptionPane.showMessageDialog(parentComponent,
                                      message,
                                      title,
                                      messageType);
    }
}
