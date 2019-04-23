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
package proguard;


/**
 * This <code>Exception</code> signals that a parse exception of some
 * sort has occurred.
 *
 * @author Eric Lafortune
 */
public class ParseException extends Exception {

    /**
     * Constructs a <code>ParseException</code> with <code>null</code>
     * as its error detail message.
     */
    public ParseException() {
        super();
    }

    /**
     * Constructs a <code>ParseException</code> with the specified detail
     * message. The error message string <code>s</code> can later be
     * retrieved by the <code>{@link Throwable#getMessage}</code>
     * method of class <code>Throwable</code>.
     *
     * @param s the detail message.
     */
    public ParseException(String s) {
        super(s);
    }
}
