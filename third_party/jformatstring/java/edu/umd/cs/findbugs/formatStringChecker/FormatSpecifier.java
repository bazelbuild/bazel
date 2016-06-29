/*
 * Copyright 2003-2007 Sun Microsystems, Inc.  All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Sun designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Sun in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Sun Microsystems, Inc., 4150 Network Circle, Santa Clara,
 * CA 95054 USA or visit www.sun.com if you need additional information or
 * have any questions.
 */
/* This file has been extensively modified from the original Sun implementation
 * to provide for compile time checking of Format Strings.
 * 
 * These modifications were performed by Bill Pugh, this code is free software.
 * 
 */
package edu.umd.cs.findbugs.formatStringChecker;

import java.io.Serializable;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.Calendar;
import java.util.Date;
import java.util.FormatFlagsConversionMismatchException;
import java.util.GregorianCalendar;
import java.util.IllegalFormatFlagsException;
import java.util.IllegalFormatPrecisionException;
import java.util.IllegalFormatWidthException;
import java.util.MissingFormatWidthException;
import java.util.UnknownFormatConversionException;

public class FormatSpecifier {

	private int index = -1;
	private Flags f = Flags.NONE;
	private int width;
	private int precision;
	private boolean dt = false;
	private char c;
	private final String source;

	public String toString() {
		return source;
	}

    private int index(String s) throws FormatterNumberFormatException {
		if (s != null) {
			try {
				index = Integer.parseInt(s.substring(0, s.length() - 1));
			} catch (NumberFormatException x) {
                throw new FormatterNumberFormatException(s, "index");
			}
		} else {
			index = 0;
		}
		return index;
	}

	public int index() {
		return index;
	}

	private Flags flags(String s) {
		f = Flags.parse(s);
		if (f.contains(Flags.PREVIOUS))
			index = -1;
		return f;
	}

	Flags flags() {
		return f;
	}

    private int width(String s) throws FormatterNumberFormatException {
		width = -1;
		if (s != null) {
			try {
				width = Integer.parseInt(s);
				if (width < 0)
					throw new IllegalFormatWidthException(width);
			} catch (NumberFormatException x) {
                throw new FormatterNumberFormatException(s, "width");
			}
		}
		return width;
	}

    private int precision(String s) throws FormatterNumberFormatException {
		precision = -1;
		if (s != null) {
			try {
				// remove the '.'
				precision = Integer.parseInt(s.substring(1));
				if (precision < 0)
					throw new IllegalFormatPrecisionException(precision);
			} catch (NumberFormatException x) {
                throw new FormatterNumberFormatException(s, "precision");
			}
		}
		return precision;
	}

	int precision() {
		return precision;
	}

	private char conversion(String s) {
		c = s.charAt(0);
		if (!dt) {
			if (!Conversion.isValid(c))
				throw new UnknownFormatConversionException(String.valueOf(c));
			if (Character.isUpperCase(c))
				f.add(Flags.UPPERCASE);
			c = Character.toLowerCase(c);
			if (Conversion.isText(c))
				index = -2;
		}
		return c;
	}

	FormatSpecifier(String source, String[] sa)
            throws FormatFlagsConversionMismatchException,
            FormatterNumberFormatException {
		int idx = 0;
		this.source = source;
		index(sa[idx++]);
		flags(sa[idx++]);
		width(sa[idx++]);
		precision(sa[idx++]);

		if (sa[idx] != null) {
			dt = true;
			if (sa[idx].equals("T"))
				f.add(Flags.UPPERCASE);
		}
		conversion(sa[++idx]);

		if (dt)
			checkDateTime();
		else if (Conversion.isGeneral(c))
			checkGeneral();
		else if (Conversion.isCharacter(c))
			checkCharacter();
		else if (Conversion.isInteger(c))
			checkInteger();
		else if (Conversion.isFloat(c))
			checkFloat();
		else if (Conversion.isText(c))
			checkText();
		else
			throw new UnknownFormatConversionException(String.valueOf(c));
	}

	private void checkGeneral() throws FormatFlagsConversionMismatchException {
		if ((c == Conversion.BOOLEAN || c == Conversion.HASHCODE)
				&& f.contains(Flags.ALTERNATE))
			failMismatch(Flags.ALTERNATE, c);
		// '-' requires a width
		if (width == -1 && f.contains(Flags.LEFT_JUSTIFY))
			throw new MissingFormatWidthException(toString());
		checkBadFlags(Flags.PLUS, Flags.LEADING_SPACE, Flags.ZERO_PAD,
				Flags.GROUP, Flags.PARENTHESES);
	}

	private void checkDateTime() throws FormatFlagsConversionMismatchException {
		if (precision != -1)
			throw new IllegalFormatPrecisionException(precision);
		if (!DateTime.isValid(c))
			throw new UnknownFormatConversionException("t" + c);
		checkBadFlags(Flags.ALTERNATE, Flags.PLUS, Flags.LEADING_SPACE,
				Flags.ZERO_PAD, Flags.GROUP, Flags.PARENTHESES);
		// '-' requires a width
		if (width == -1 && f.contains(Flags.LEFT_JUSTIFY))
			throw new MissingFormatWidthException(toString());
	}

	private void checkCharacter() throws FormatFlagsConversionMismatchException {
		if (precision != -1)
			throw new IllegalFormatPrecisionException(precision);
		checkBadFlags(Flags.ALTERNATE, Flags.PLUS, Flags.LEADING_SPACE,
				Flags.ZERO_PAD, Flags.GROUP, Flags.PARENTHESES);
		// '-' requires a width
		if (width == -1 && f.contains(Flags.LEFT_JUSTIFY))
			throw new MissingFormatWidthException(toString());
	}

	private void checkInteger() throws FormatFlagsConversionMismatchException {
		checkNumeric();
		if (precision != -1)
			throw new IllegalFormatPrecisionException(precision);

		if (c == Conversion.DECIMAL_INTEGER)
			checkBadFlags(Flags.ALTERNATE);
		else
			checkBadFlags(Flags.GROUP);
	}

	private void checkBadFlags(Flags... badFlags)
			throws FormatFlagsConversionMismatchException {
		for (int i = 0; i < badFlags.length; i++)
			if (f.contains(badFlags[i]))
				failMismatch(badFlags[i], c);
	}

	private void checkFloat() throws FormatFlagsConversionMismatchException {
		checkNumeric();
		if (c == Conversion.DECIMAL_FLOAT) {
		} else if (c == Conversion.HEXADECIMAL_FLOAT) {
			checkBadFlags(Flags.PARENTHESES, Flags.GROUP);
		} else if (c == Conversion.SCIENTIFIC) {
			checkBadFlags(Flags.GROUP);
		} else if (c == Conversion.GENERAL) {
			checkBadFlags(Flags.ALTERNATE);
		}
	}

	private void checkNumeric() {
		if (width != -1 && width < 0)
			throw new IllegalFormatWidthException(width);

		if (precision != -1 && precision < 0)
			throw new IllegalFormatPrecisionException(precision);

		// '-' and '0' require a width
		if (width == -1
				&& (f.contains(Flags.LEFT_JUSTIFY) || f
						.contains(Flags.ZERO_PAD)))
			throw new MissingFormatWidthException(toString());

		// bad combination
		if ((f.contains(Flags.PLUS) && f.contains(Flags.LEADING_SPACE))
				|| (f.contains(Flags.LEFT_JUSTIFY) && f
						.contains(Flags.ZERO_PAD)))
			throw new IllegalFormatFlagsException(f.toString());
	}

	private void checkText() {
		if (precision != -1)
			throw new IllegalFormatPrecisionException(precision);
		switch (c) {
		case Conversion.PERCENT_SIGN:
			if (f.valueOf() != Flags.LEFT_JUSTIFY.valueOf()
					&& f.valueOf() != Flags.NONE.valueOf())
				throw new IllegalFormatFlagsException(f.toString());
			// '-' requires a width
			if (width == -1 && f.contains(Flags.LEFT_JUSTIFY))
				throw new MissingFormatWidthException(toString());
			break;
		case Conversion.LINE_SEPARATOR:
			if (width != -1)
				throw new IllegalFormatWidthException(width);
			if (f.valueOf() != Flags.NONE.valueOf())
				throw new IllegalFormatFlagsException(f.toString());
			break;
		default:
            throw new UnknownFormatConversionException(String.valueOf(c));
		}
	}

	public void print(String arg, int argIndex)
			throws IllegalFormatConversionException,
			FormatFlagsConversionMismatchException {

		try {
			if (arg.charAt(0) == '[')

				failConversion(arg);

			if (dt) {
				printDateTime(arg);
				return;
			}
			switch (c) {
			case Conversion.DECIMAL_INTEGER:
			case Conversion.OCTAL_INTEGER:
			case Conversion.HEXADECIMAL_INTEGER:
				printInteger(arg);
				break;
			case Conversion.SCIENTIFIC:
			case Conversion.GENERAL:
			case Conversion.DECIMAL_FLOAT:
			case Conversion.HEXADECIMAL_FLOAT:
				printFloat(arg);
				break;
			case Conversion.CHARACTER:
			case Conversion.CHARACTER_UPPER:
				printCharacter(arg);
				break;
			case Conversion.BOOLEAN:
				printBoolean(arg);
				break;
			case Conversion.STRING:
			case Conversion.HASHCODE:
			case Conversion.LINE_SEPARATOR:
			case Conversion.PERCENT_SIGN:
				break;
			default:
                throw new UnknownFormatConversionException(String.valueOf(c));
			}
		} catch (IllegalFormatConversionException e) {
			e.setArgIndex(argIndex);
			throw e;
		}
	}

	private boolean matchSig(String signature, Class<?>... classes) {
		for (Class<?> c : classes)
			if (matchSig(signature, c))
				return true;
		return false;
	}

	private boolean matchSig(String signature, Class<?> c) {
		return signature.equals("L" + c.getName().replace('.', '/') + ";");
	}

	private boolean mightBeUnknown(String arg) {
		if (matchSig(arg, Object.class) || matchSig(arg, Number.class)
				|| matchSig(arg, Serializable.class)
				|| matchSig(arg, Comparable.class))
			return true;
		return false;
	}

	private void printInteger(String arg)
			throws IllegalFormatConversionException,
			FormatFlagsConversionMismatchException {
		if (mightBeUnknown(arg))
			return;
		if (matchSig(arg, Byte.class, Short.class, Integer.class, Long.class))
			printLong();
		else if (matchSig(arg, BigInteger.class)) {
		} else
			failConversion(arg);
	}

	private void printFloat(String arg) throws IllegalFormatConversionException {
		if (mightBeUnknown(arg))
			return;
		if (matchSig(arg, Float.class, Double.class)) {
		} else if (matchSig(arg, BigDecimal.class)) {
			printBigDecimal(arg);
		} else
			failConversion(arg);
	}

	private void printDateTime(String arg)
			throws IllegalFormatConversionException {
		if (mightBeUnknown(arg))
			return;
        if (matchSig(arg, Long.class, Date.class, java.sql.Date.class,
                java.sql.Time.class, java.sql.Timestamp.class, Calendar.class,
                GregorianCalendar.class)) {
		} else {
			failConversion(arg);
		}

	}

	private void printCharacter(String arg)
			throws IllegalFormatConversionException {
		if (mightBeUnknown(arg))
			return;
		if (matchSig(arg, Character.class, Byte.class, Short.class,
				Integer.class)) {
		} else {
			failConversion(arg);
		}

	}

	private void printBoolean(String arg)
			throws IllegalFormatConversionException {
		if (mightBeUnknown(arg))
			return;
		if (matchSig(arg, Boolean.class))
			return;
		failConversion(arg);
	}

	private void printLong() throws FormatFlagsConversionMismatchException {

		if (c == Conversion.OCTAL_INTEGER) {
			checkBadFlags(Flags.PARENTHESES, Flags.LEADING_SPACE, Flags.PLUS);
		} else if (c == Conversion.HEXADECIMAL_INTEGER) {
			checkBadFlags(Flags.PARENTHESES, Flags.LEADING_SPACE, Flags.PLUS);
		}

	}

	private void printBigDecimal(String arg)
			throws IllegalFormatConversionException {
		if (c == Conversion.HEXADECIMAL_FLOAT)
			failConversion(arg);
	}

	private void failMismatch(Flags f, char c)
			throws FormatFlagsConversionMismatchException {
		String fs = f.toString();
		throw new FormatFlagsConversionMismatchException(fs, c);
	}

	private void failConversion(String arg)
			throws IllegalFormatConversionException {
        if (dt)
            throw new IllegalFormatConversionException(this.toString(),
                    f.contains(Flags.UPPERCASE) ? 'T' : 't', arg);
		throw new IllegalFormatConversionException(this.toString(), c, arg);
	}

}