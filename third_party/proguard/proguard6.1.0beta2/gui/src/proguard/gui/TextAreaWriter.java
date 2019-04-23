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
import java.io.*;


/**
 * This writer appends its output to a given text area.
 *
 * @author Eric Lafortune
 */
final class TextAreaWriter extends FilterWriter implements Runnable
{
    private final JTextArea textArea;


    public TextAreaWriter(JTextArea textArea)
    {
        super(new CharArrayWriter());

        this.textArea = textArea;
    }


    // Implementation for FilterWriter.

    public void flush() throws IOException
    {
        super.flush();

        try
        {
            // Append the accumulated buffer contents to the text area.
            SwingUtil.invokeAndWait(this);
        }
        catch (Exception e)
        {
            // Nothing.
        }
    }


    // Implementation for Runnable.

    public void run()
    {
        CharArrayWriter writer = (CharArrayWriter)super.out;

        // Has any new text been written?
        String text = writer.toString();
        if (text.length() > 0)
        {
            // Append the accumulated text to the text area.
            textArea.append(text);

            // Clear the buffer.
            writer.reset();
        }
    }
}
