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

import java.util.ArrayList;
import java.util.FormatFlagsConversionMismatchException;
import java.util.IllegalFormatException;
import java.util.List;
import java.util.UnknownFormatConversionException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class Formatter {

	public static void check(String format, String... args)
			throws ExtraFormatArgumentsException,
			IllegalFormatConversionException, IllegalFormatException,
			FormatFlagsConversionMismatchException,
            MissingFormatArgumentException, FormatterNumberFormatException {

		// index of last argument referenced
		int last = -1;
		// last ordinary index
		int lasto = -1;

		// last index used
		int maxIndex = -1;

		for (FormatSpecifier fs : parse(format)) {
			int index = fs.index();
			switch (index) {
			case -2:
				// ignore it
				break;
			case -1: // relative index
				if (last < 0 || last > args.length - 1)
					throw new MissingFormatArgumentException(last,
							fs.toString());
				fs.print(args[last], last);
				break;
			case 0: // ordinary index
				lasto++;
				last = lasto;
				if (lasto > args.length - 1)
					throw new MissingFormatArgumentException(lasto,
							fs.toString());
				maxIndex = Math.max(maxIndex, lasto);
				fs.print(args[lasto], lasto);
				break;
			default: // explicit index
				last = index - 1;
				if (last > args.length - 1)
					throw new MissingFormatArgumentException(last,
							fs.toString());
				maxIndex = Math.max(maxIndex, last);
				fs.print(args[last], last);
				break;
			}

		}
		if (maxIndex < args.length - 1) {
			throw new ExtraFormatArgumentsException(args.length, maxIndex + 1);
		}
	}

	// %[argument_index$][flags][width][.precision][t]conversion
	private static final String formatSpecifier = "%(\\d+\\$)?([-#+ 0,(\\<]*)?(\\d+)?(\\.\\d+)?([tT])?([a-zA-Z%])";

	private static Pattern fsPattern = Pattern.compile(formatSpecifier);

	// Look for format specifiers in the format string.
	private static List<FormatSpecifier> parse(String s)
            throws FormatFlagsConversionMismatchException,
            FormatterNumberFormatException {
		ArrayList<FormatSpecifier> al = new ArrayList<FormatSpecifier>();
		Matcher m = fsPattern.matcher(s);
		int i = 0;
		while (i < s.length()) {
			if (m.find(i)) {
				// Anything between the start of the string and the beginning
				// of the format specifier is either fixed text or contains
				// an invalid format string.
				if (m.start() != i) {
					// Make sure we didn't miss any invalid format specifiers
					checkText(s.substring(i, m.start()));
				}

				// Expect 6 groups in regular expression
				String[] sa = new String[6];
				for (int j = 0; j < m.groupCount(); j++) {
					sa[j] = m.group(j + 1);
				}
				al.add(new FormatSpecifier(m.group(0), sa));
				i = m.end();
			} else {
				// No more valid format specifiers. Check for possible invalid
				// format specifiers.
				checkText(s.substring(i));
				// The rest of the string is fixed text
				break;

			}
		}

		return al;
	}

	private static void checkText(String s) {
		int idx;
		// If there are any '%' in the given string, we got a bad format
		// specifier.
		if ((idx = s.indexOf('%')) != -1) {
			char c = (idx > s.length() - 2 ? '%' : s.charAt(idx + 1));
			throw new UnknownFormatConversionException(String.valueOf(c));
		}
	}
}
