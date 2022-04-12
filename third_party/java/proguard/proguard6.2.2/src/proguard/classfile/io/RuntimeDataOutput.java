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
package proguard.classfile.io;

import java.io.*;

/**
 * This class delegates its method calls to the corresponding DataOutput methods,
 * converting its IOExceptions to RuntimeExceptions.
 *
 * The class provides two convenience methods that additionally check whether the
 * written values are unsigned resp. signed short values before writing them.
 *
 * @author Eric Lafortune
 */
final class RuntimeDataOutput
{
    private final DataOutput dataOutput;


    public RuntimeDataOutput(DataOutput dataOutput)
    {
        this.dataOutput = dataOutput;
    }


    // Methods delegating to DataOutput.

    public void write(byte[] b)
    {
        try
        {
            dataOutput.write(b);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }


    public void write(byte[] b, int off, int len)
    {
        try
        {
            dataOutput.write(b, off, len);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }


    public void write(int b)
    {
        try
        {
            dataOutput.write(b);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }


    public void writeBoolean(boolean v)
    {
        try
        {
            dataOutput.writeBoolean(v);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }


    public void writeByte(int v)
    {
        try
        {
            dataOutput.writeByte(v);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }


    public void writeBytes(String s)
    {
        try
        {
            dataOutput.writeBytes(s);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }


    public void writeChar(int v)
    {
        try
        {
            dataOutput.writeChar(v);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }


    public void writeChars(String s)
    {
        try
        {
            dataOutput.writeChars(s);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }


    public void writeDouble(double v)
    {
        try
        {
            dataOutput.writeDouble(v);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }


    public void writeFloat(float v)
    {
        try
        {
            dataOutput.writeFloat(v);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }


    public void writeInt(int v)
    {
        try
        {
            dataOutput.writeInt(v);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }


    public void writeLong(long v)
    {
        try
        {
            dataOutput.writeLong(v);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }


    /**
     * Checks if the given value is an unsigned short value before writing it.
     *
     * @throws IllegalArgumentException if the value is not an unsigned short value.
     * @see #writeShort(int)
     */
    public void writeUnsignedShort(int v)
    {
        if ((v & 0xffff) != v)
        {
            throw new IllegalArgumentException("Overflow of unsigned short value ["+v+"]");
        }

        writeShort(v);
    }


    /**
     * Checks if the given value is a signed short value before writing it.
     *
     * @throws IllegalArgumentException if the value is not a signed short value.
     * @see #writeShort(int)
     */
    public void writeSignedShort(int v)
    {
        if ((short)v != v)
        {
            throw new IllegalArgumentException("Overflow of signed short value ["+v+"]");
        }

        writeShort(v);
    }


    public void writeShort(int v)
    {
        try
        {
            dataOutput.writeShort(v);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }


    public void writeUTF(String str)
    {
        try
        {
            dataOutput.writeUTF(str);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }
}
