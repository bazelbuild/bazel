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

import java.text.MessageFormat;
import java.util.ResourceBundle;


/**
 * This class provides some utility methods for working with resource bundles.
 *
 * @author Eric Lafortune
 */
class GUIResources
{
    private static final ResourceBundle messages  = ResourceBundle.getBundle(GUIResources.class.getName());
    private static final MessageFormat  formatter = new MessageFormat("");


    /**
     * Returns an internationalized message, based on its key.
     */
    public static String getMessage(String messageKey)
    {
        return messages.getString(messageKey);
    }


    /**
     * Returns an internationalized, formatted message, based on its key, with
     * the given arguments.
     */
    public static String getMessage(String messageKey, Object[] messageArguments)
    {
        formatter.applyPattern(messages.getString(messageKey));
        return formatter.format(messageArguments);
    }
}
