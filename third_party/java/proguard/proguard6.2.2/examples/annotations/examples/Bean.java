import proguard.annotation.*;

/**
 * This bean illustrates the use of annotations for configuring ProGuard.
 *
 * You can compile it with:
 *     javac -classpath ../lib/annotations.jar Bean.java
 * You can then process it with:
 *     java -jar ../../../lib/proguard.jar @ ../examples.pro
 *
 * The annotations will preserve the class and its public getters and setters,
 * as a result of the specifications in lib/annotations.pro.
 */
@Keep
@KeepPublicGettersSetters
public class Bean
{
    public boolean booleanProperty;
    public int     intProperty;
    public String  stringProperty;


    public boolean isBooleanProperty()
    {
        return booleanProperty;
    }


    public void setBooleanProperty(boolean booleanProperty)
    {
        this.booleanProperty = booleanProperty;
    }


    public int getIntProperty()
    {
        return intProperty;
    }


    public void setIntProperty(int intProperty)
    {
        this.intProperty = intProperty;
    }


    public String getStringProperty()
    {
        return stringProperty;
    }


    public void setStringProperty(String stringProperty)
    {
        this.stringProperty = stringProperty;
    }
}
